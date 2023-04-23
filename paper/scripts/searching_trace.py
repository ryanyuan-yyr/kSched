input_path = r'data\RTX2080Ti\va_1_240_mm_1_61_mannually_correct_for_range_va_4_61_mm_4_61'

Z = list()

with open(input_path) as input:
    for line in input:
        Z.append([float(time) for time in line.split()][4:61])

Z = Z[4:61]

min_max = [0, 0]

for i, op in enumerate([min, max]):
    min_max[i] = op(value for line in Z for value in line)

min_max = tuple(min_max)


def linear_map(value, range: tuple, target_range: tuple):
    return max(target_range[0], min(int(target_range[0] + (value - range[0])/(range[1]-range[0])*(target_range[1]-target_range[0])), target_range[1]))


class Identifier:
    def __init__(self, init, step=1) -> None:
        self.id = init
        self.step = step

    def use(self):
        res = self.id
        self.id += self.step
        return res


width = 10

with open(r'paper\figures\searching_trace\searching_trace_va_mm.drawio', 'w') as o:
    id = Identifier(2)
    o.write(
        '''
    <mxfile host="">
    <diagram id="" name="Page-1">
        <mxGraphModel dx="" dy="" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="" pageHeight="" math="0" shadow="0">
            <root>
                <mxCell id="0"/>
                <mxCell id="1" parent="0"/>
        '''
    )

    for i, line in enumerate(Z):
        for j, value in enumerate(line):
            x = width*j
            y = width*i
            min_color = 0xff, 0xff, 0xff
            max_color = 0x00, 0x99, 0x00
            r, g, b = linear_map(value, min_max, (max_color[0], min_color[0])), linear_map(
                value, min_max, (max_color[1], min_color[1])), linear_map(value, min_max, (max_color[2], min_color[2]))
            o.write(
                f'''
                                <mxCell id="{id.use()}" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#{r:02x}{g:02x}{b:02x};fontColor=#ffffff;strokeColor=none;gradientColor=none;fillStyle=auto;" vertex="1" parent="1">
                    <mxGeometry x="{x}" y="{y}" width="{width}" height="{width}" as="geometry"/>
                </mxCell>
                '''
            )

    searching_trace = [
        (9, 44),
        (29, 44),
        (29, 24),
        (49, 24),
        (49, 4),
        (59, 4),
        (60, 4),
    ]

    searching_trace = [(point[0]-4, point[1]-4) for point in searching_trace]

    [start_config, *inner_configs, target_config] = searching_trace

    def config2axis(config):
        return config[1]*width + width//2, config[0]*width + width//2

    start_axes, target_axes = config2axis(
        start_config), config2axis(target_config)
    o.write(
        f'''
                <mxCell id="{id.use()}" value="" style="endArrow=classic;html=1;rounded=1;endSize=8;startSize=8;edgeStyle=orthogonalEdgeStyle;strokeWidth=4;startArrow=oval;startFill=1;" parent="1" edge="1">
                    <mxGeometry width="50" height="50" relative="1" as="geometry">
                        <mxPoint x="{start_axes[0]}" y="{start_axes[1]}" as="sourcePoint"/>
                        <mxPoint x="{target_axes[0]}" y="{target_axes[1]}" as="targetPoint"/>
                        <Array as="points">
        '''
    )

    for inner_config in inner_configs:
        inner_axes = config2axis(inner_config)
        o.write(f'''        
            <mxPoint x="{inner_axes[0]}" y="{inner_axes[1]}"/>
        ''')

    o.write(
        f'''
                        </Array>
                    </mxGeometry>
                </mxCell>
                <mxCell id="{id.use()}" value="" style="edgeStyle=segmentEdgeStyle;endArrow=classic;html=1;curved=0;rounded=0;endSize=8;startSize=8;strokeColor=#000000;strokeWidth=4;startArrow=classic;startFill=1;" parent="1" edge="1">
                    <mxGeometry width="50" height="50" relative="1" as="geometry">
                        <mxPoint x="640" y="-40" as="sourcePoint"/>
                        <mxPoint x="-40" y="640" as="targetPoint"/>
                    </mxGeometry>
                </mxCell>
                <mxCell id="{id.use()}" value="Size of &lt;font face=&quot;Courier New&quot;&gt;matrix_mul&lt;/font&gt; subkernel" style="text;strokeColor=none;fillColor=none;html=1;fontSize=24;fontStyle=1;verticalAlign=middle;align=center;fontFamily=Times New Roman;" parent="1" vertex="1">
                    <mxGeometry x="460" y="-120" width="330" height="70" as="geometry"/>
                </mxCell>
                <mxCell id="{id.use()}" value="Size of &lt;font face=&quot;Courier New&quot;&gt;vec_add&lt;/font&gt; subkernel" style="text;strokeColor=none;fillColor=none;html=1;fontSize=24;fontStyle=1;verticalAlign=middle;align=center;fontFamily=Times New Roman;rotation=270;" parent="1" vertex="1">
                    <mxGeometry x="-250" y="550" width="330" height="70" as="geometry"/>
                </mxCell>
        '''
    )

    o.write(
        '''
                    </root>
        </mxGraphModel>
    </diagram>
</mxfile>
        '''
    )
