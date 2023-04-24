import matplotlib.pyplot as plt
import json


class Kernel:
    def __init__(self, short_name_idx, short_name) -> None:
        self.short_name_idx = short_name_idx
        self.short_name = short_name
        self.subkernel_list = []

    def push_subkernel(self, start, end):
        self.subkernel_list.append((start, end - start))

    def get_short_name(self):
        return self.short_name

    def get_subkernel_list(self):
        return self.subkernel_list

    def get_subkernel_name_idx(self):
        return self.short_name_idx
    
    def for_each(self, map):
        for i in range(len(self.get_subkernel_list())):
            self.get_subkernel_list()[i] = map(self.get_subkernel_list()[i])


kernel_list = [Kernel(62, 'vec_add'), Kernel(65, 'matrix_mul')]
with open(r'paper\figures\bad_cosched\va_1_mm_32_worst.json') as f:
    for line in f:
        subkernel_json = json.loads(line)
        subkernel_start, subkernel_end = float(
            subkernel_json['CudaEvent']['startNs'])/1e9, float(subkernel_json['CudaEvent']['endNs'])/1e9
        subkernel_shortname_idx = int(
            subkernel_json['CudaEvent']['kernel']['shortName'])
        for kernel in kernel_list:
            if kernel.get_subkernel_name_idx() == subkernel_shortname_idx:
                kernel.push_subkernel(subkernel_start, subkernel_end)
                break
        else:
            print("Unknown kernel")
            exit()

starting_time = min(map(lambda kernel: kernel.get_subkernel_list()[0][0], kernel_list))

for kernel in kernel_list:
    kernel.for_each(lambda subkernel: (subkernel[0] - starting_time, subkernel[1]))

with open(r'paper\figures\bad_cosched\timeline.drawio', 'w') as o:
    # 1. prescript
    o.write(
        '''<mxfile host="65bd71144e">
    <diagram id="" name="Page-1">
        <mxGraphModel dx="2104" dy="2347" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="827" pageHeight="1169" math="0" shadow="0">
            <root>
                <mxCell id="0"/>
                <mxCell id="1" parent="0"/>
        '''
    )

    color_list = ['#fad7ac', '#b0e3e6']
    height = 100
    id = 2
    scale = 1e6/2

    # 2. Kernels
    for (i, kernel), color in zip(enumerate(kernel_list), color_list): 
        name = kernel.get_short_name()
        for j, (subkernel_start, subkernel_dur) in enumerate(kernel.get_subkernel_list()):
            x = subkernel_start * scale
            y = i*height
            width = subkernel_dur * scale
            o.write(
                f'''                <mxCell id="{id}" value="" style="rounded=1;whiteSpace=wrap;html=1;fontSize=51;fontFamily=Times New Roman;fillColor={color};strokeColor=none;fontColor=#000000;" parent="1" vertex="1">
                        <mxGeometry x="{x}" y="{y}" width="{width}" height="{height}" as="geometry"/>
                    </mxCell>
                '''
            )
            id += 1

    # 3. postscript
    o.write(
        '''
            </root>
        </mxGraphModel>
    </diagram>
</mxfile>'''
    )

