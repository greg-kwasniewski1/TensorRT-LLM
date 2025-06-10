import os
import re


def recolor(
        nodes: dict,
        dir: str = '/mnt/c/greg_stuff/code/TensorRT-LLM/examples/auto_deploy',
        svg_filename: str = "deepseek_ep_shard.svg"):
    """
    Set node colors in svg file
    Inputs:
        nodes: dict, a dictionary {color[hex]: list[node_name]}
        dir: str, the directory to save the svg file
        svg_filename: str, the name of the svg file
    Outputs:
        None
    """

    # Node format in svg:
    #     <!-- add_9 -->
    # <g id="node191" class="node">
    # <title>add_9</title>
    # <path fill="#00ff0e" stroke=" (...)/>

    # Look for nodes where <title>node_name</title> matches a node_name in the nodes dict
    # and replace the fill color with the color in the nodes dict
    with open(os.path.join(dir, svg_filename), 'r') as f:
        svg_content = f.read()

    for color, node_names in nodes.items():
        for node_name in node_names:
            # use regex to find the node
            # the original node color can be arbitrary:
            pattern = f'<title>{node_name}</title>\n<path fill="([^"]*)"'
            # find all matches
            re.findall(pattern, svg_content)
            svg_content = re.sub(
                pattern, f'<title>{node_name}</title>\n<path fill="{color}"',
                svg_content)

    with open(os.path.join(dir, svg_filename), 'w') as f:
        f.write(svg_content)


if __name__ == "__main__":
    start_color = "#00ff00"
    end_color = "#0000ff"
    unaccounted_color = "#ff0000"
    nodes = {
        start_color: [],
        end_color: [
            "mean_1", "mean_2", "mean_3", "mean_4", "mean_5", "mean_6",
            "mean_7", "mean_8"
        ],
        unaccounted_color: []
    }
    recolor(nodes)
