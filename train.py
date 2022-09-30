#!/usr/bin/env python3

import os
import json
import typer
import torch
import pathlib
import rich.box
import rich.rule
import numpy as np
import torch.nn as nn
from enum import Enum
import rich.terminal_theme
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.columns import Columns
from torchvision import transforms
from collections import OrderedDict
from torch.utils.data import DataLoader
from typing import Dict, Iterable, Literal
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as F
from rich.console import Console, RenderableType, Group
from rich.progress import Progress, ProgressColumn, TextColumn, BarColumn, TimeRemainingColumn, SpinnerColumn, track

# custom imports
from mobilevit import MobileViT
from helpers import PrettyPrintingHelpers

# typer app
app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
console = Console(record=True)


# custom progress classes
class CustomProgress(Progress):
    class CompletedColumn(ProgressColumn):

        def render(self, task):
            """Calculate common unit for completed and total."""
            download_status = f"{int(task.completed)} of {int(task.total)}"
            return Text(download_status, style="progress.download")


    class TransferSpeedColumn(ProgressColumn):
        """Renders human readable transfer speed."""


        def render(self, task):
            """Show data transfer speed."""
            speed = task.speed
            if speed is None:
                return Text("?", style="progress.data.speed")
            speed = f"{speed:,.{2}f}"
            return Text(f"{speed} B/s", style="progress.data.speed")


    def __init__(self, *args, use_info_table: bool = True, rich_console: Console = None, **kwargs):
        super(CustomProgress, self).__init__(*args, **kwargs)

        self.info_table = Table(show_footer=False, box=rich.box.HORIZONTALS, style="dim grey39", row_styles=["dim", ""])
        self.info_table.add_column("PHASE", justify="right", style="bold")

        self.train_style = "dark_sea_green4"
        self.test_style = "medium_purple1"
        self.use_info_table = use_info_table
        self.info_table_rows = list()
        self.rich_console = rich_console


    def generate_table(self):
        """
        Regenerate table to counter the table going past visible section of the screen.
        """

        # layout for checking rendering issues
        layout = Layout()

        # recreate table
        info_table = Table(show_footer=False, box=rich.box.HORIZONTALS, style="dim grey39", row_styles=["dim", ""])

        # add columns
        for column in self.info_table.columns:
            info_table.add_column(column.header, style=column.style, justify=column.justify)

        # get terminal size
        number_of_displayable_rows = os.get_terminal_size()[1]
        number_of_displayable_rows -= 6

        # add rows dynamically
        while number_of_displayable_rows >= 0:
            # recreate table
            info_table = Table(show_footer=False, box=rich.box.HORIZONTALS, style="dim grey39", row_styles=["dim", ""])

            # add columns
            for column in self.info_table.columns:
                info_table.add_column(column.header, style=column.style, justify=column.justify)

            # add rows
            for row_inx, table_row in enumerate(self.info_table_rows[-number_of_displayable_rows:]):
                if row_inx == 0 and table_row["row_end_section"]:
                    continue

                # append row
                info_table.add_row(
                        *table_row["row_data"],
                        style=table_row["row_style"],
                        end_section=table_row["row_end_section"],
                )

            # update layout and check if rendering issues exist
            layout.update(info_table)
            render_map = layout.render(self.rich_console, self.rich_console.options)
            if len(render_map[layout].render[-1]) > 2:
                number_of_displayable_rows -= 4
            else:
                break

        # reset table
        self.info_table = info_table

        return None


    def add_info_table_cols(self, new_cols):
        """
        Add cols from ordered dict if not present in info_table
        """

        cols = set([x.header for x in self.info_table.columns])
        missing = set(new_cols) - cols
        if len(missing) == 0:
            return

        # iterate on new_cols since they are in order
        for c in new_cols:
            if c in missing and c != "checkpointed":
                self.info_table.add_column(c, justify="left")


    def update_info_table(self, aux: Dict[str, float], phase: str):
        """
        Update the info_table with the latest results
        :param aux:
        :param phase: either 'train' or 'test'
        """

        self.add_info_table_cols(aux.keys())
        epoch = aux.pop("epoch")
        checkpointed = aux.pop("checkpointed")
        aux = OrderedDict((k, f"{v:>4.05f}") for k, v in aux.items())
        if phase == "Train":
            st = self.train_style
        else:
            st = self.test_style

        # append row to data
        self.info_table_rows.append(
                {
                    # 'row_data':        [phase, f"{epoch}{' ' if checkpointed else ''}", *list(aux.values())],
                    'row_data':        [phase, f"{epoch}{' :trophy:' if checkpointed else ''}", *list(aux.values())],
                    'row_style':       st,
                    'row_end_section': (phase != "Train"),
                }
        )


    def get_renderables(self) -> Iterable[RenderableType]:
        """Display progress together with info table"""

        # this method is called once before the init, so check if the attribute is present
        if hasattr(self, "use_info_table"):
            use_table = self.use_info_table
            info_table = self.info_table
        else:
            use_table = False
            info_table = Table()

        if use_table:
            # generate table
            self.generate_table()
            info_table = self.info_table

            task_table = self.make_tasks_table(self.tasks)
            renderable = Columns((info_table, task_table), align="left", expand=True)
        else:
            renderable = self.make_tasks_table(self.tasks)

        yield renderable


class ProgressBarLogger(object):
    """
    Displays a progress bar with information about the current epoch and the epoch progression.
    """


    def __init__(
            self,
            n_epochs: int,
            console: Console,
            train_data_len: int = 0,
            test_data_len: int = 0,
            use_info_table: bool = True,
    ):
        """
        :param n_epochs: total number of epochs
        :param train_data_len: length of the dataset generation for training
        :param test_data_len: length of the dataset generation for testing
        :param use_info_table: true to add an information table on top of the progress bar
        """

        self.n_epochs = n_epochs
        self.train_data_len = train_data_len
        self.test_data_len = test_data_len
        self.progress = CustomProgress(
                TextColumn(
                        "[bold yellow]EPOCH {task.fields[cur_epoch]} of {task.fields[n_epochs]} [dim]•[/dim] [blue]{task.fields[mode]}",
                        justify="right",
                ),
                SpinnerColumn(spinner_name="dots", speed=1.2),
                BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "[dim]•",
                CustomProgress.CompletedColumn(),
                "[dim]•",
                CustomProgress.TransferSpeedColumn(),
                "[dim]•",
                TimeRemainingColumn(),
                use_info_table=use_info_table,
                rich_console=console,
        )

        self.progress.start()
        self.train_p = self.progress.add_task(
                description="",
                mode="Train",
                cur_epoch=0,
                n_epochs=self.n_epochs,
                start=False,
                visible=False,
                total=self.train_data_len,
        )
        self.test_p = self.progress.add_task(
                description="",
                mode="Validation",
                cur_epoch=0,
                n_epochs=self.n_epochs,
                start=False,
                visible=False,
                total=self.test_data_len,
        )


    @staticmethod
    def build_od(loss, epoch, checkpointed: bool = False):
        od = OrderedDict()
        od["epoch"] = epoch
        od["loss"] = loss
        od["checkpointed"] = checkpointed

        return od


    def on_epoch_begin(self, epoch: int):
        self.progress.reset(
                task_id=self.train_p,
                total=self.train_data_len,
                start=False,
                visible=False,
                cur_epoch=epoch,
                n_epochs=self.n_epochs,
                mode="Train",
        )
        self.progress.start_task(self.train_p)
        self.progress.update(self.train_p, visible=True)


    def on_epoch_end(self, loss: float, epoch: int):
        self.progress.stop_task(self.train_p)
        self.progress.update(self.train_p, visible=False)

        # if the datalen is zero update with the one epoch just ended
        if self.train_data_len == 0:
            self.train_data_len = self.progress.tasks[self.train_p].completed

        self.progress.reset(
                task_id=self.train_p,
                total=self.train_data_len,
                start=False,
                visible=False,
                cur_epoch=epoch,
                n_epochs=self.n_epochs,
                mode="Train",
        )

        od = self.build_od(loss, epoch)
        self.progress.update_info_table(od, "Train")


    def on_test_begin(self, epoch: int):
        self.progress.reset(
                task_id=self.test_p,
                total=self.test_data_len,
                start=False,
                visible=False,
                cur_epoch=epoch,
                n_epochs=self.n_epochs,
                mode="Validation",
        )

        self.progress.start_task(self.test_p)
        self.progress.update(self.test_p, visible=True)


    def on_test_end(self, loss: float, epoch: int, checkpointed: bool):
        self.progress.stop_task(self.test_p)
        self.progress.update(self.test_p, visible=False)

        # if the datalen is zero update with the one epoch just ended
        if self.test_data_len == 0:
            self.test_data_len = self.progress.tasks[self.test_p].completed

        self.progress.reset(
                task_id=self.test_p,
                total=self.test_data_len,
                start=False,
                visible=False,
                cur_epoch=epoch,
                n_epochs=self.n_epochs,
                mode="Validation",
        )

        od = self.build_od(loss, epoch, checkpointed=checkpointed)
        self.progress.update_info_table(od, "Validation")


    def on_train_end(self):
        self.progress.stop()


    def on_batch_end(
            self, loss: float, batch_id: int, is_training: bool = True
    ):
        if is_training:
            self.progress.update(self.train_p, refresh=True, advance=1)
        else:
            self.progress.update(self.test_p, refresh=True, advance=1)


# enums
class MobileViTModelSize(Enum):
    xxs = "XXS"
    xs = "XS"
    s = "S"


class MobileViTExportFormat(Enum):
    onnx = "ONNX"
    torchscript = "Torchscript"


class MobileViTExportDeviceFormat(Enum):
    cpu = "CPU"
    gpu = "GPU"


# typer functions
@app.command('check')
def check_if_working():
    """Check if MobileViT is working as expected with a random Torch sample."""

    # generate random sample
    img = torch.randn(5, 3, 256, 256)

    # run XXS format
    vit = mobilevit_xxs()
    out = vit(img)
    console.print()
    console.rule(title="[bold red]MobileViT XXS")
    console.print(f"[bold yellow]Output Shape:[/] [dim white]{out.shape}")
    console.print(f"[bold cyan]Model Parameter Count:[/] [dim white]{count_parameters(vit):,}")

    # run XS format
    vit = mobilevit_xs()
    out = vit(img)
    console.print()
    console.rule(title="[bold red]MobileViT XS")
    console.print(f"[bold yellow]Output Shape[/]: [dim white]{out.shape}")
    console.print(f"[bold cyan]Model Parameter Count:[/] [dim white]{count_parameters(vit):,}")

    # run S format
    vit = mobilevit_s()
    out = vit(img)
    console.print()
    console.rule(title="[bold red]MobileViT S")
    console.print(f"[bold yellow]Output Shape[/]: [dim white]{out.shape}")
    console.print(f"[bold cyan]Model Parameter Count:[/] [dim white]{count_parameters(vit):,}")
    console.print()


class LetterBoxResize(object):
    """Perform letterbox resizing on input image."""


    def __init__(self, max_width: int, max_height: int, fill_color: int = 0, padding_type: str = "constant"):
        super(LetterBoxResize, self).__init__()
        self.max_width = max_width
        self.max_height = max_height
        self.fill_color = fill_color
        self.padding_type = padding_type


    def __call__(self, image):
        """Perform resizing."""

        # gather current image dimensions
        image_width, image_height = image.size

        # calculate aspect ratio's
        max_aspect_ratio = self.max_width / self.max_height
        image_aspect_ratio = image_width / image_height

        # check if aspect ratio is close to same
        if round(max_aspect_ratio, 2) != round(image_aspect_ratio, 2):
            # calculate padding to preserve aspect ratio
            height_padding = int(image_width / max_aspect_ratio - image_height)
            width_padding = int(image_height * max_aspect_ratio - image_width)

            if height_padding > 0 and width_padding < 0:
                height_padding //= 2
                return F.resize(F.pad(image, (0, height_padding, 0, height_padding), self.fill_color, self.padding_type), [self.max_height, self.max_width])

            else:
                width_padding //= 2
                return F.resize(F.pad(image, (width_padding, 0, width_padding, 0), self.fill_color, self.padding_type), [self.max_height, self.max_width])

        return F.resize(image, [self.max_height, self.max_width])


@app.command('train', no_args_is_help=True)
def train_mobilevit_classification_model(
        dataset_dir: pathlib.Path = typer.Option(..., '-i', help="Dataset directory path (dataset should be structured as shown above)."),
        checkpoints_savepath: pathlib.Path = typer.Option(..., '-s', help="Directory path for saving model weights."),
        image_dim: int = typer.Option(..., '--imgsize', help="Image dimensions (note: training is done with square images, hence single integer entry.)."),
        batch_size: int = typer.Option(..., '--batchsize', help="Batch size for training."),
        epochs: int = typer.Option(..., '--epochs', help="Number of epochs of training."),
        dataloader_workers: int = typer.Option(4, '--dataloader_workers', help="Number of dataloader workers."),
        mobilevit_size: MobileViTModelSize = typer.Option(MobileViTModelSize.s.value, '--model-size', show_choices=True, show_default=True, help="Choice of MobileViT model size.", case_sensitive=False),
        learning_rate: float = typer.Option(0.001, '--lr', help="Learning Rate.", show_default=True),
        momentum: float = typer.Option(0.9, '--momentum', help="Learning Momentum.", show_default=True),
        early_stopping: int = typer.Option(12, '--early-stop-after', help="Number of epochs before early stopping."),
        use_letterbox_resizing: bool = typer.Option(True, '--no-letterbox', help="Apply LetterBox resizing on images?"),
):
    """
    Train MobileViT model on a custom dataset.

    \b
    dataset_dir
    |--- train
        |--- img_1.png
        |--- img_2.png
        |--- ...
    |--- valid
        |--- img_1.png
        |--- img_2.png
        |--- ...
    |--- [test] optional
        ...
    """

    # create 'checkpoints_savepath' directory if doesn't already exist
    checkpoints_savepath.mkdir(parents=True, exist_ok=True)
    model_checkpoint_savepath = checkpoints_savepath / "best.pt"

    # build image transforms objects
    train_transforms = transforms.Compose(
            [
                LetterBoxResize(max_width=image_dim, max_height=image_dim, fill_color=0, padding_type="constant") if use_letterbox_resizing else transforms.Resize((image_dim, image_dim)),
                transforms.ToTensor(),
            ]
    )
    valid_transforms = transforms.Compose(
            [
                LetterBoxResize(max_width=image_dim, max_height=image_dim, fill_color=0, padding_type="constant") if use_letterbox_resizing else transforms.Resize((image_dim, image_dim)),
                transforms.ToTensor(),
            ]
    )

    # generate train, validation and test directory paths
    train_dir = dataset_dir / "train"
    valid_dir = dataset_dir / "valid"
    test_dir = dataset_dir / "test"
    if not test_dir.exists():
        test_dir = None

    # gather datasets
    train_set = ImageFolder(train_dir, train_transforms)
    valid_set = ImageFolder(valid_dir, valid_transforms)
    if test_dir is not None:
        test_set = ImageFolder(test_dir, valid_transforms)

    # determine number of classes
    number_of_classes = len(train_set.classes)

    # generate dataloaders
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers)
    valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers)
    if test_dir is not None:
        test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers)

    # save class2idx mapping
    idx_to_class = {v: k for k, v in train_set.class_to_idx.items()}
    with open(checkpoints_savepath / "class_to_idx.json", "w") as f:
        json.dump(train_set.class_to_idx, f, indent=3)

    # determine model dims and channels
    if mobilevit_size == MobileViTModelSize.xxs:
        model_dims = [64, 80, 96]
        model_channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    elif mobilevit_size == MobileViTModelSize.xs:
        model_dims = [96, 120, 144]
        model_channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    else:
        model_dims = [144, 192, 240]
        model_channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]

    # define MobileViT model
    model = MobileViT(
            image_size=(image_dim, image_dim),
            dims=model_dims,
            channels=model_channels,
            num_classes=number_of_classes,
            expansion=4 if mobilevit_size != MobileViTModelSize.xxs else 2,
            kernel_size=3,
            patch_size=(2, 2),
    )

    # define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # move model to GPU
    training_device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(training_device)

    # create progress logger
    print()
    with console.screen():
        progress_logger = ProgressBarLogger(
                n_epochs=epochs,
                train_data_len=len(train_dataloader),
                test_data_len=len(valid_dataloader),
                use_info_table=True,
                console=console,
        )

        # start model training
        no_improvement_epochs = 0
        prev_validation_loss = np.inf
        for epoch in range(epochs):  # loop over the dataset multiple times
            # track epoch start
            progress_logger.on_epoch_begin(epoch=epoch + 1)

            # run training
            running_loss = 0.0
            for batch_inx, batch_data in enumerate(train_dataloader, 1):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = batch_data
                inputs, labels = inputs.to(training_device), labels.to(training_device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # gather running loss
                running_loss += loss.item()

                # track batch end
                progress_logger.on_batch_end(loss=loss.item(), batch_id=batch_inx, is_training=True)

            # track epoch end
            progress_logger.on_epoch_end(loss=running_loss, epoch=epoch + 1)

            # run validation after every epoch
            model.eval()
            with torch.no_grad():
                # track validation test start
                progress_logger.on_test_begin(epoch=epoch + 1)

                validation_loss = 0.0
                for batch_inx, batch_data in enumerate(valid_dataloader, 1):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = batch_data
                    inputs, labels = inputs.to(training_device), labels.to(training_device)

                    # forward
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    validation_loss += loss.item()

                    # track batch end
                    progress_logger.on_batch_end(loss=loss.item(), batch_id=batch_inx, is_training=False)

                # save checkpoint, if better than prev. validation loss
                checkpointed = False
                if validation_loss <= prev_validation_loss:
                    prev_validation_loss = validation_loss
                    torch.save(model.state_dict(), model_checkpoint_savepath)
                    checkpointed = True
                    no_improvement_epochs = 0
                else:
                    no_improvement_epochs += 1

                # track validation test end
                progress_logger.on_test_end(loss=validation_loss, epoch=epoch + 1, checkpointed=checkpointed)

            # early stopping
            if no_improvement_epochs >= early_stopping:
                break

            # put model to train mode
            model.train()

        # gather model training logs table
        training_logs_table = Table(
                show_footer=progress_logger.progress.info_table.show_footer,
                box=progress_logger.progress.info_table.box,
                style=progress_logger.progress.info_table.style,
                row_styles=progress_logger.progress.info_table.row_styles,
        )
        for column in progress_logger.progress.info_table.columns:
            training_logs_table.add_column(column.header, style=column.style, justify=column.justify)
        for table_row in progress_logger.progress.info_table_rows:
            # append row
            training_logs_table.add_row(
                    *table_row["row_data"],
                    style=table_row["row_style"],
                    end_section=table_row["row_end_section"],
            )

        # track training end
        progress_logger.on_train_end()

    # print training metrics table
    console.print(training_logs_table)
    console.print()

    # acknowledge if early stopped
    if epoch + 1 != epochs:
        console.print(f":raised_hand: [bold medium_orchid]Early Stopped at EPOCH {epoch + 1}")
        console.print()

    # if testset is present, load the best weights and check model performance on the testset.
    if test_dir is not None:
        # delete existing model
        model.cpu()
        del model

        # create new model
        model = MobileViT(
                image_size=(image_dim, image_dim),
                dims=model_dims,
                channels=model_channels,
                num_classes=number_of_classes,
                expansion=4 if mobilevit_size != MobileViTModelSize.xxs else 2,
                kernel_size=3,
                patch_size=(2, 2),
        )

        # load best model weights
        console.print(f":electric_plug: [bold deep_sky_blue4]Loading Checkpoint:[/] [dim white]{model_checkpoint_savepath.resolve().__str__()}")
        model.load_state_dict(torch.load(model_checkpoint_savepath))
        console.print()

        # move model to correct training device
        model.to(training_device)

        # put model to evaluation mode
        model.eval()

        # test model performance
        overall_correct = 0
        test_stats = {c: {ic: 0 for ic in sorted(train_set.class_to_idx.keys())} for c in sorted(train_set.class_to_idx.keys())}
        with torch.no_grad():
            for batch_data in track(sequence=test_dataloader, description="[dark_magenta]Testset Performance", show_speed=True, transient=True, style="dim dark_magenta"):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = batch_data
                inputs, labels = inputs.to(training_device), labels.to(training_device)

                # forward
                outputs = model(inputs)

                # get max scoring class
                _, predicted = torch.max(outputs.data, 1)

                # iterate over each image result and collect stats
                for i in range(labels.size(0)):
                    # move to cpu
                    actual_inx, predicted_inx = labels[i].cpu().item(), predicted[i].cpu().item()
                    test_stats[idx_to_class[actual_inx]][idx_to_class[predicted_inx]] += 1
                    overall_correct += int(predicted_inx == actual_inx)

        # print test performance stats
        total_images = len(test_set)
        PrettyPrintingHelpers.pretty_classification_stats(
                model_title=f"MobileViT ({mobilevit_size.value.upper()} Arch)",
                dataset_name=dataset_dir.name.capitalize(),
                total_count=total_images,
                correct_count=overall_correct,
                server_issue_class=None,
                classes=list(test_stats.keys()),
                accuracy_stats=test_stats,
        )

        # save stats SVG
        console.save_svg(path=(checkpoints_savepath / "console_log.svg").resolve().__str__(), theme=rich.terminal_theme.MONOKAI, title=f"Console Log for MobileViT ({mobilevit_size.value.upper()} Arch) Training on {dataset_dir.name.capitalize()} Dataset with Image Dimension of {image_dim}px")
        console.print(f"[bold deep_sky_blue4]:floppy_disk: Saved Report:[/] [dim white]{(checkpoints_savepath / 'console_log.svg').resolve().__str__()}")
        print()

        # print test performance stats
        # total_images = len(test_set)
        # overall_accuracy = (overall_correct / total_images) * 100
        # confusion_matrix_header_layout = Columns(
        #         [
        #             Panel.fit(renderable="[dim]Actual", subtitle=":point_down:", border_style="magenta"),
        #             Panel.fit(renderable="[dim]Predicted", title=":point_right:", border_style="cyan"),
        #         ], align='center', expand=True, equal=True,
        # )
        # rtable = Table(title=f"[bold dim yellow]Classification Accuracy Stats", caption=f"[turquoise2][b]Overall Model Accuracy: {overall_accuracy:02.01f} %[/b]  [dim white]({overall_correct} of {total_images})", box=rich.box.HEAVY_HEAD, show_lines=True, expand=False, style="dim grey39")
        # rtable.add_column(header=confusion_matrix_header_layout, justify='right', no_wrap=True, style="bold dark_magenta", max_width=27, vertical="middle")
        # for cls in test_stats:
        #     rtable.add_column(header=f"[bold dim cyan]{cls.upper()}", justify='center', no_wrap=True, vertical="middle", style="sky_blue3")
        #
        # for cls in test_stats:
        #     row_items = list()
        #     for inner_cls in test_stats:
        #         current_count = test_stats[cls][inner_cls]
        #         if cls == inner_cls:
        #             item_str = f"[dark_olive_green3][b]{current_count:2d}[/b]"
        #         else:
        #             item_str = f"[b]{current_count:2d}[/b]"
        #         item_accuracy_str = f"[steel_blue]{(current_count / sum(test_stats[cls].values())) * 100:02.01f}%"
        #
        #         # append row item
        #         panel_border_style = "dim steel_blue"
        #         if cls == inner_cls:
        #             panel_border_style = "yellow"
        #
        #         row_items.append(
        #                 Panel(
        #                         renderable=Group(
        #                                 item_str,
        #                                 rich.rule.Rule(align='center', style='dim grey39'),
        #                                 item_accuracy_str,
        #                         ),
        #                         style=panel_border_style,
        #                         expand=True,
        #                 )
        #         )
        #
        #     # append row
        #     rtable.add_row(
        #             cls.upper(),
        #             *row_items,
        #     )
        #
        # # with console.pager(styles=True, links=True):
        # #     console.print(rtable)
        #
        # console.print(rtable)
        # console.print()

    # save console logs
    console.save_html(path=(checkpoints_savepath / "console_log.html").resolve().__str__(), theme=rich.terminal_theme.MONOKAI)
    console.print(f"[bold deep_sky_blue4]:floppy_disk: Saved Report:[/] [dim white]{(checkpoints_savepath / 'console_log.html').resolve().__str__()}")
    print()


@app.command('export', no_args_is_help=True)
def export_mobilevit_model_weight(
        model_weight: pathlib.Path = typer.Option(..., '-i', help="Model weight filepath."),
        export_format: MobileViTExportFormat = typer.Option(..., '-f', show_choices=True, help="Export format.", case_sensitive=False),
        image_dim: int = typer.Option(..., '--imgsize', help="Image dimensions (note: training is done with square images, hence single integer entry.)."),
        dataset_dir: pathlib.Path = typer.Option(..., '-d', help="Dataset directory path (dataset should be structured as shown above)."),
        mobilevit_size: MobileViTModelSize = typer.Option(..., '--model-size', show_choices=True, help="Choice of MobileViT model size.", case_sensitive=False),
        training_device: MobileViTExportDeviceFormat = typer.Option(MobileViTExportDeviceFormat.cpu.value, '--device', show_choices=True, help="Target device.", case_sensitive=False),
):
    """
    Export MobileViT model weights to different supported formats.

    \b
    dataset_dir
    |--- train
        |--- img_1.png
        |--- img_2.png
        |--- ...
    |--- valid
        |--- img_1.png
        |--- img_2.png
        |--- ...
    |--- [test] optional
        ...
    """

    # determine number of classes
    number_of_classes = len([c for c in (dataset_dir / "train").iterdir() if c.is_dir()])

    # determine model dims and channels
    if mobilevit_size == MobileViTModelSize.xxs:
        model_dims = [64, 80, 96]
        model_channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    elif mobilevit_size == MobileViTModelSize.xs:
        model_dims = [96, 120, 144]
        model_channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    else:
        model_dims = [144, 192, 240]
        model_channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]

    # define MobileViT model
    model = MobileViT(
            image_size=(image_dim, image_dim),
            dims=model_dims,
            channels=model_channels,
            num_classes=number_of_classes,
            expansion=4 if mobilevit_size != MobileViTModelSize.xxs else 2,
            kernel_size=3,
            patch_size=(2, 2),
    )

    # load checkpoint weight
    console.print()
    console.print(f":electric_plug: [bold deep_sky_blue4]Loading Checkpoint:[/] [dim white]{model_weight.resolve().__str__()}")
    model.load_state_dict(torch.load(model_weight))
    console.print()

    # move model to correct training device
    model.to(training_device.value.lower())

    # put model to evaluation mode
    model.eval()

    # create sample image
    example_image = torch.rand(1, 3, image_dim, image_dim, requires_grad=True).to(training_device.value.lower())

    # traced model savepath
    traced_model_savepath = (model_weight.parent / f"{model_weight.with_suffix('').name}_{training_device.value.lower()}.{export_format.value.lower()}").resolve().__str__()

    # export to requested format
    if export_format == MobileViTExportFormat.onnx:
        # trace ONNX model
        torch.onnx.export(
                model,
                example_image,
                traced_model_savepath,
                export_params=True,
                opset_version=10,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
        )

    elif export_format == MobileViTExportFormat.torchscript:
        # trace model
        traced_model = torch.jit.trace(model, example_image)

        # save traced model
        traced_model.save(traced_model_savepath)

    # print log
    console.print(f":floppy_disk: [bold deep_sky_blue4]Exported Checkpoint ({export_format.value.lower()}):[/] [dim white]{traced_model_savepath}")
    console.print()


# run app
app()
