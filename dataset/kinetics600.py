"""Random-access K600 TAR clips, sampled by timestamp at the requested FPS."""
import fcntl
import io
import json
import math
from pathlib import Path
import random
import tarfile
import warnings

import av
import torch
from torchvision.transforms.v2 import functional as VF


class ShortVideoError(ValueError):
    pass


def build_shard_index(shards):
    """Index uncompressed TAR byte ranges without extracting video files."""
    samples = []
    for shard_id, path in enumerate(shards):
        records = {}
        with tarfile.open(path, 'r:') as archive:
            for member in archive:
                if not member.isfile():
                    continue
                key, ext = member.name.rsplit('.', 1)
                if ext not in {'mp4', 'cls', 'json'}:
                    continue
                record = records.setdefault(key, {'key': key, 'shard': shard_id})
                if ext == 'mp4':
                    record.update(offset=member.offset_data, size=member.size)
                elif ext == 'cls':
                    record['label'] = int(archive.extractfile(member).read().strip())
                else:
                    metadata = json.load(archive.extractfile(member))
                    if 'class_id' in metadata:
                        record['label'] = int(metadata['class_id'])
        for record in records.values():
            if 'offset' not in record:
                continue
            if 'label' not in record or not 0 <= record['label'] < 600:
                raise ValueError(f'Missing/invalid class label in {path}: {record["key"]}')
            samples.append(record)
    if not samples:
        raise ValueError('No labelled MP4 samples found in the selected shards')
    return samples


def decode_clip(data, num_frames, fps, resolution, random_start=True):
    """Select the first source frame at/after each 1/fps timestamp.

    Timestamp sampling avoids the FPS error from rounding native_fps / fps to
    an integer stride (e.g. sampling a 30-fps source every 8 frames).
    """
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'SLICE'
        stream.codec_context.thread_count = 1
        source_fps = float(stream.average_rate or stream.guessed_rate or 0)
        if source_fps < fps - 0.01:
            raise ValueError(f'Source FPS {source_fps} is below requested FPS {fps}')
        if stream.duration is None:
            raise ValueError('Video has no stream duration for timestamp sampling')
        duration = float(stream.duration * stream.time_base)
        span = (num_frames - 1) / fps
        max_start = duration - 1 / source_fps - span
        if max_start < -1e-4:
            raise ShortVideoError(f'Video duration {duration:.3f}s is too short for {num_frames} frames at {fps} fps')
        max_start_frame = max(0, math.floor(max_start * source_fps + 1e-6))
        start_frame = random.randint(0, max_start_frame) if random_start else max_start_frame // 2
        start = start_frame / source_fps
        origin = float((stream.start_time or 0) * stream.time_base)
        targets = [origin + start + i / fps for i in range(num_frames)]
        container.seek(int(targets[0] / stream.time_base), stream=stream, backward=True)
        frames = []
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * stream.time_base)
            if timestamp + 1e-7 < targets[len(frames)]:
                continue
            frames.append(torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2, 0, 1))
            if len(frames) == num_frames:
                break
        if len(frames) != num_frames:
            raise ShortVideoError(f'Only decoded {len(frames)}/{num_frames} requested timestamps')
    video = torch.stack(frames)
    if min(video.shape[-2:]) < resolution:
        raise ValueError(f'Source is only {tuple(video.shape[-2:])}; refusing to upscale to {resolution}px')
    if min(video.shape[-2:]) != resolution:
        video = VF.resize(video, resolution, antialias=True)
    video = VF.center_crop(video, [resolution, resolution])
    return video.float().div_(255).contiguous()


class Kinetics600VideoDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, current_epoch=0):
        self.cfg = cfg
        self.shards = sorted(Path(cfg.save_dir).expanduser().glob(cfg.shard_pattern))
        if not self.shards:
            raise FileNotFoundError(f'No {cfg.shard_pattern} in {cfg.save_dir}; set K600_ROOT to the training shard directory')
        if cfg.frame_skip != 1:
            raise ValueError('K600 uses timestamp-based fps sampling; frame_skip must be 1')
        signature = [[str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns] for p in self.shards]
        cache = Path(cfg.index_cache)
        cache.parent.mkdir(parents=True, exist_ok=True)
        # All DDP ranks share one cache; serialize creation and replace atomically.
        with cache.with_suffix('.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cached = json.loads(cache.read_text()) if cache.exists() else None
            if cached is None or cached.get('signature') != signature:
                cached = {'signature': signature, 'samples': build_shard_index(self.shards)}
                temp = cache.with_suffix('.tmp')
                temp.write_text(json.dumps(cached))
                temp.replace(cache)
        self.samples = cached['samples']
        self.set_epoch(current_epoch)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # A few native K600 videos are shorter than four seconds. Resample them
        # explicitly; retain the replacement clip's own label.
        for attempt in range(16):
            sample = self.samples[index]
            with self.shards[sample['shard']].open('rb') as f:
                f.seek(sample['offset'])
                data = f.read(sample['size'])
            try:
                video = decode_clip(data, self.cfg.max_frames, self.cfg.fps, self.cfg.resolution)
            except ShortVideoError as exc:
                if attempt == 0:
                    warnings.warn(f'Resampling short K600 clip {sample["key"]}: {exc}')
                index = random.randrange(len(self))
                continue
            return {'videos': video, 'conds': torch.tensor(sample['label'], dtype=torch.long)}
        raise RuntimeError('Could not find a sufficiently long K600 clip in 16 attempts')
