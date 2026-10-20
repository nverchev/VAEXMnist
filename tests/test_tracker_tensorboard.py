"""Tests for the TensorBoard tracker writing images."""

import pathlib

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.config import Experiment, get_experiment_config
from src.trackers.tensorboard import ImageTensorBoard
from src.lib.events import ImageEvent, ImageGridEvent


def _read_tags(tmp_path: pathlib.Path) -> dict[str, list[str]]:
    run_dir = next(path.parent for path in tmp_path.rglob('events.out.tfevents.*'))
    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    tags = accumulator.Tags()
    return {'images': sorted(tags['images']), 'text': sorted(tags['tensors'])}


def test_a_grid_and_its_captions_are_written(tmp_path: pathlib.Path) -> None:
    """A grid with uneven rows is written as one image, its captions as text, under its folder."""
    exp = Experiment(get_experiment_config(), par_dir=tmp_path)
    exp.trackers.subscribe(ImageTensorBoard())
    image = torch.rand(1, 8, 8)
    with exp.create_run():
        ImageGridEvent(images=[[image, image], [image]], captions=[['a', 'b'], ['c']], title='grid', folder='sample_0')

    tags = _read_tags(tmp_path)

    assert tags['images'] == ['sample_0/grid']
    assert tags['text'] == ['sample_0/grid/text_summary']


def test_a_single_image_is_written(tmp_path: pathlib.Path) -> None:
    """A single image is written under its title."""
    exp = Experiment(get_experiment_config(), par_dir=tmp_path)
    exp.trackers.subscribe(ImageTensorBoard())
    with exp.create_run():
        ImageEvent(image=torch.rand(3, 8, 8), title='single')

    tags = _read_tags(tmp_path)

    assert tags['images'] == ['single']
    assert tags['text'] == []


def test_an_unrecorded_run_writes_no_image(tmp_path: pathlib.Path) -> None:
    """An image published in an unrecorded run is not written."""
    exp = Experiment(get_experiment_config(), par_dir=tmp_path)
    exp.trackers.subscribe(ImageTensorBoard())
    with exp.create_run(record=False):
        ImageEvent(image=torch.rand(3, 8, 8), title='single')

    written = list(tmp_path.rglob('events.out.tfevents.*'))

    assert written == []
