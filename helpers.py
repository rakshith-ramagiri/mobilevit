#!/usr/bin/env python3

import rich.box
import rich.rule
import rich.terminal_theme
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.columns import Columns
from typing import Dict, List, Union
from rich.console import Console, RenderableType, Group
from rich.progress import Progress, SpinnerColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn, BarColumn, ProgressColumn


class PrettyPrintingHelpers(object):

    @staticmethod
    def pretty_classification_stats(
            model_title: str,
            dataset_name: str,
            total_count: int,
            correct_count: int,
            classes: List[str],
            accuracy_stats: Dict[str, Dict[str, Union[int, List[str]]]],
            server_issue_class: Union[str, None],
            average_inference_speed: Union[float, None],

    ) -> str:
        """Pretty print classification accuracy stats."""

        # define console
        console = Console(record=True)

        console.print()
        overall_accuracy = (correct_count / total_count) * 100
        confusion_matrix_header_layout = Columns(
                [
                    Panel.fit(renderable="[dim]Actual", subtitle=":point_down:", border_style="magenta"),
                    Panel.fit(renderable="[dim]Predicted", title=":point_right:", border_style="cyan"),
                ], align='center', expand=True, equal=True,
        )
        rtable = Table(title=f"[bold yellow]Classification Accuracy Stats", caption=f"[turquoise2][b]Overall Model Accuracy: {overall_accuracy:02.01f} %[/b]  [white]({correct_count} of {total_count})  [indian_red](avg. inference time: {(average_inference_speed or 0):>02.1f} ms)", box=rich.box.HORIZONTALS, show_lines=True, expand=False, style="dim grey39", show_edge=True)
        rtable.add_column(header=confusion_matrix_header_layout, justify='right', no_wrap=True, style="bold magenta", max_width=27, vertical="middle")
        for cls in classes:
            rtable.add_column(header=f"[bold cyan]{cls.upper()}", justify='center', no_wrap=True, vertical="middle", style="sky_blue3")
        if server_issue_class is not None:
            rtable.add_column(header=f"[bold red]{server_issue_class.upper()}", justify='center', no_wrap=True, vertical="middle", style="dim red")

        for cls in classes:
            row_items = list()
            for inner_cls in classes + [server_issue_class]:
                if inner_cls is None:
                    continue

                current_count = len(accuracy_stats[cls][inner_cls]) if isinstance(accuracy_stats[cls][inner_cls], list) else accuracy_stats[cls][inner_cls]
                if cls == inner_cls:
                    item_str = f"[dark_olive_green3][b]{current_count:2d}[/b]"
                else:
                    item_str = f"[b]{current_count:2d}[/b]"
                item_accuracy_str = f"[steel_blue]{(current_count / sum(len(e) for e in accuracy_stats[cls].values())) * 100:02.01f}%"

                # append row item
                panel_border_style = "dim steel_blue"
                if inner_cls == server_issue_class:
                    panel_border_style = "dim red"
                elif cls == inner_cls:
                    panel_border_style = "yellow"

                row_items.append(
                        Panel(
                                renderable=Group(
                                        item_str,
                                        rich.rule.Rule(align='center', style='dim grey39'),
                                        item_accuracy_str,
                                ),
                                style=panel_border_style,
                                expand=True,
                        )
                )

            # append row
            rtable.add_row(
                    cls.upper(),
                    *row_items,
            )

        console.print(rtable)
        console.print()

        return console.export_svg(theme=rich.terminal_theme.MONOKAI, title=f"{model_title} on {dataset_name}")
