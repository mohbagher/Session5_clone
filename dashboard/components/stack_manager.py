"""
Stack Manager Component
=======================
Experiment stack UI and management logic.
"""

import ipywidgets as widgets


def create_stack_manager():
    """Create experiment stack manager component."""

    # Custom experiment name input
    widget_custom_exp_name = widgets.Text(
        value='',
        placeholder='Enter custom experiment name (optional)',
        description='Custom Name:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%'),
        tooltip='Leave empty for auto-generated name'
    )

    # Stack display (list of experiments)
    widget_stack_display = widgets.Select(
        options=[],
        description='Experiment Stack:',
        layout=widgets.Layout(width='100%', height='150px'),
        tooltip='Select experiments to remove or reorder'
    )

    # Stack control buttons
    button_add_to_stack = widgets.Button(
        description='ADD TO STACK',
        button_style='primary',
        layout=widgets.Layout(width='140px'),
        icon='plus',
        tooltip='Add current configuration to stack'
    )

    button_remove_from_stack = widgets.Button(
        description='REMOVE',
        button_style='danger',
        layout=widgets.Layout(width='120px'),
        icon='minus',
        tooltip='Remove selected experiment from stack'
    )

    button_move_up = widgets.Button(
        description='UP',
        button_style='info',
        layout=widgets.Layout(width='100px'),
        icon='arrow-up',
        tooltip='Move selected experiment up'
    )

    button_move_down = widgets.Button(
        description='DOWN',
        button_style='info',
        layout=widgets.Layout(width='100px'),
        icon='arrow-down',
        tooltip='Move selected experiment down'
    )

    button_clear_stack = widgets.Button(
        description='CLEAR ALL',
        button_style='warning',
        layout=widgets.Layout(width='120px'),
        icon='trash',
        tooltip='Clear entire stack'
    )

    button_save_stack = widgets.Button(
        description='SAVE STACK',
        button_style='success',
        layout=widgets.Layout(width='130px'),
        icon='save',
        tooltip='Save stack configuration to file'
    )

    button_load_stack = widgets.Button(
        description='LOAD STACK',
        button_style='warning',
        layout=widgets.Layout(width='130px'),
        icon='folder-open',
        tooltip='Load stack configuration from file'
    )

    button_run_stack = widgets.Button(
        description='RUN STACK',
        button_style='success',
        layout=widgets.Layout(width='100%', height='60px'),
        icon='rocket',
        tooltip='Run all experiments in stack sequentially'
    )

    button_pause_training = widgets.Button(
        description='PAUSE',
        button_style='warning',
        layout=widgets.Layout(width='100px'),
        icon='pause',
        tooltip='Pause stack execution'
    )

    button_resume_training = widgets.Button(
        description='RESUME',
        button_style='success',
        layout=widgets.Layout(width='100px'),
        icon='play',
        tooltip='Resume stack execution'
    )

    # Layout
    stack_layout = widgets.VBox([
        widgets.HTML("<b>Experiment Stack:</b> Configure parameters above, name (optional), then add to stack."),
        widget_custom_exp_name,
        widgets.HBox([
            button_add_to_stack,
            button_remove_from_stack,
            button_move_up,
            button_move_down
        ], layout=widgets.Layout(justify_content='flex-start', margin='5px 0')),
        widget_stack_display,
        widgets.HBox([
            button_save_stack,
            button_load_stack,
            button_clear_stack
        ], layout=widgets.Layout(justify_content='flex-start', margin='5px 0')),
        widgets.HBox([
            button_run_stack
        ], layout=widgets.Layout(margin='5px 0')),
        widgets.HBox([
            button_pause_training,
            button_resume_training
        ], layout=widgets.Layout(justify_content='flex-start', margin='5px 0'))
    ], layout=widgets.Layout(
        border='2px solid #673AB7',
        padding='15px',
        margin='10px 0',
        background_color='#f3e5f5'
    ))

    # Store widget references
    stack_layout._widgets = {
        'custom_exp_name': widget_custom_exp_name,
        'stack_display': widget_stack_display,
        'button_add_to_stack': button_add_to_stack,
        'button_remove_from_stack': button_remove_from_stack,
        'button_move_up': button_move_up,
        'button_move_down': button_move_down,
        'button_clear_stack': button_clear_stack,
        'button_save_stack': button_save_stack,
        'button_load_stack': button_load_stack,
        'button_run_stack': button_run_stack,
        'button_pause_training': button_pause_training,
        'button_resume_training': button_resume_training
    }

    return stack_layout
