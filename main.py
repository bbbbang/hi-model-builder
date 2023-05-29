import dearpygui.dearpygui as dpg

import demo

import cv2

import importlib
import numpy as np
from warnings import warn
from PIL import Image

import glob
import shutil
import os
import sys

import re

def save_callback():
    print("Save Clicked")

def node_link_callback(sender, app_data):
    print("node link : ", sender, app_data)
    _node_link_item = dpg.add_node_link(app_data[0], app_data[1], parent=sender)

    sender_info = dpg.get_item_info(sender)
    sender_info[f"{_node_link_item}"] = [app_data[0], app_data[1]]

    dpg.set_item_user_data(_node_link_item, (app_data[0], app_data[1]))

    prev_node = dpg.get_item_children(app_data[0])
    next_node = dpg.get_item_children(app_data[1])

    for k, v in prev_node.items():
        if len(v) > 0:
            val = dpg.get_value(v[0])
            prev_item = v[0]

    for k, v in next_node.items():
        if len(v) > 0:
            dpg.set_value(v[0], val)
            next_item = v[0]

    dpg.configure_item(next_item, source=prev_item)

def node_delink_callback(sender, app_data):
    print("node delink : ", sender, app_data)

    _prev, _next = dpg.get_item_user_data(app_data)

    prev_node = dpg.get_item_children(_prev)
    next_node = dpg.get_item_children(_next)

    for k, v in next_node.items():
        if len(v) > 0:
            print("asdf", dpg.get_item_source(v[0]))
            #dpg.configure_item(v[0], source=0)

    dpg.delete_item(app_data)

    # prev_node = dpg.get_item_children(app_data[0])
    # next_node = dpg.get_item_children(app_data[1])

    # for k, v in prev_node.items():
    #     if len(v) > 0:
    #         prev_item = v[0]

    # for k, v in next_node.items():
    #     if len(v) > 0:
    #         next_item = v[0]



def button_callback():
    print(dpg.get_viewport_height())


def add_node(sender, data, user_data):

    print(user_data)

    window_id = user_data["id"]
    node_type = user_data["type"]
    node_count = user_data[node_type]

    if node_type == "input":
        with dpg.node(label=f"input {node_count}", pos=[500, 10], parent=window_id, tag=f"input_{node_count}") as _new_node:
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_input_int(label="width", width=200)
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_input_int(label="height", width=200)
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_input_int(label="channel", width=200)

    elif node_type == "conv2d":
        with dpg.node(label=f"conv2d {node_count}", pos=[500, 10], parent=window_id, tag=f"conv2d_{node_count}") as _new_node:
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_input_int(label="in_channels", width=200)
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_input_int(label="out_channels", width=200)
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_input_int(label="kernel", width=200, default_value=1)
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_input_int(label="padding", width=200, default_value=0)
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_input_int(label="stride", width=200, default_value=1)
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_input_int(label="dilation", width=200, default_value=1)
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_input_int(label="groups", width=200, default_value=1)

    elif node_type == "batchnorm":
        with dpg.node(label=f"batchnorm {node_count}", pos=[500, 10], parent=window_id, tag=f"batchnorm_{node_count}") as _new_node:
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                bn_source = dpg.add_input_int(label="in_channels", width=200)
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_input_int(label="out_channels", width=200, source=bn_source, readonly=True)

    elif node_type == "relu":
        with dpg.node(label=f"relu {node_count}", pos=[500, 10], parent=window_id, tag=f"relu_{node_count}") as _new_node:
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input):
                relu_source = dpg.add_input_int(label="in_channels", width=200)
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_input_int(label="out_channels", width=200, source=relu_source, readonly=True)

    # with dpg.item_handler_registry(show=False, tag="__node_handler"):
    #     dpg.add_item_double_clicked_handler(0, callback=lambda s, a, u: print(f"clicked_handler: {s} '\t' {a} '\t' {u}"))
    # dpg.bind_item_handler_registry(_new_node, "__node_handler")

    user_data[node_type] = user_data[node_type] + 1




class ToolWindow:
    def __init__(self, size, pos=(0,0), label: str="Tool Window", tag: str="tool_window", *args, **kwargs):
        
        self.window_width, self.window_height = size
        self.window_x, self.window_y = pos

        self.label = label
        self.tag = tag

        self.node_window_id = kwargs["node_window_id"]

        self.node_count = dict(input=0, conv2d=0, batchnorm=0, relu=0)

        self.setup()

    def setup(self):
        # tool window
        with dpg.window(label=self.label, pos=(self.window_x, self.window_y), width=self.window_width, height=self.window_height, tag=self.tag):
            with dpg.child_window(width=500, height=320, menubar=True):
                dpg.add_input_float(label="asd", width=200)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Export", callback=button_callback)
                    dpg.add_button(label="Save", callback=button_callback)
                    dpg.add_button(label="Open", callback=button_callback)
                # dpg.add_button(label="add new", callback=add_node, user_data=window_id)
            with dpg.child_window(width=500, height=320):
                dpg.add_button(label="Add Input", callback=add_node, user_data=dict(id=self.node_window_id, type="input", **self.node_count))
                dpg.add_button(label="Add Conv2d", callback=add_node, user_data=dict(id=self.node_window_id, type="conv2d", **self.node_count))
                dpg.add_button(label="Add Batchnorm", callback=add_node, user_data=dict(id=self.node_window_id, type="batchnorm", **self.node_count))
                dpg.add_button(label="Add ReLU", callback=add_node, user_data=dict(id=self.node_window_id, type="relu", **self.node_count))

    def __call__(self):
        pass


class NodeWindow:
    def __init__(self, size, pos, label: str="Node Window", tag: str="node_window", *args, **kwargs):
        
        self.window_width, self.window_height = size
        self.window_x, self.window_y = pos

        self.label = label
        self.tag = tag

        self.setup()
        
    def setup(self):
        # node window
        with dpg.window(label=self.label, pos=(self.window_x, self.window_y), width=self.window_width, height=self.window_height, tag=self.tag):
            with dpg.group(horizontal=True):
                with dpg.node_editor(callback=node_link_callback, 
                                delink_callback=node_delink_callback,
                                minimap=True, minimap_location=dpg.mvNodeMiniMap_Location_BottomRight) as window_id:
                    self.window_id = window_id

    def get_window_id(self):
        return self.window_id
    
    def __call__(self):
        pass



class AttributeWindow:
    def __init__(self, size, pos, label: str="Attr Window", tag: str="attr_window", *args, **kwargs):
        
        self.window_width, self.window_height = size
        self.window_x, self.window_y = pos

        self.label = label
        self.tag = tag

        self.node_window_id = kwargs["node_window_id"]

        self.setup()

    def setup(self):
        # attribute window
        with dpg.window(label=self.label, pos=(self.window_x, self.window_y), width=self.window_width, height=self.window_height, tag=self.tag):
            with dpg.child_window(width=500, height=320, menubar=True):
                dpg.add_input_float(label="asd", width=200)
                dpg.add_button(label="S123ave", callback=button_callback)
                #dpg.add_button(label="add new", callback=add_node, user_data=dict(id=self.node_window_id, type="input"))

    def __call__(self):
        pass


def main():

    viewport_width_offset = 15
    viewport_height_offset = 40
    viewport_width = 1920
    viewport_height = 1080

    tool_window_width = (viewport_width-viewport_width_offset)//10*2
    node_window_width = (viewport_width-viewport_width_offset)//10*6
    attr_window_width = (viewport_width-viewport_width_offset)//10*2

    tool_window_height = viewport_height - viewport_height_offset
    node_window_height = viewport_height - viewport_height_offset
    attr_window_height = viewport_height - viewport_height_offset

    dpg.create_context()
    dpg.create_viewport(width=viewport_width, height=viewport_height, x_pos=0, y_pos=0, resizable=False)
    dpg.setup_dearpygui()

    demo.show_demo()

    node_window = NodeWindow(size=(node_window_width, node_window_height), pos=(tool_window_width, 0), label="Node Window", tag="node_window")

    node_window_id = node_window.get_window_id()
    tool_window = ToolWindow(size=(tool_window_width, tool_window_height), pos=(0, 0), label="Tool Window", tag="tool_window", **{"node_window_id":node_window_id})
    attr_window = AttributeWindow(size=(attr_window_width, attr_window_height), pos=(tool_window_width+node_window_width, 0), label="Attr Window", tag="attr_window", **{"node_window_id":node_window_id})




    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__=="__main__":

    main()