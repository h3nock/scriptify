from typing import Union
import numpy as np
from xml.etree import ElementTree
from pathlib import Path


def get_writerID(file_path: Union[str, Path]) -> int:
    """
    Reads a xml file and returns the writerID from general element (tag)
    """
    writerID = 0 
    file_path = Path(file_path) 
    if file_path.exists() and file_path.is_file(): 
        try:
            xml_tree = ElementTree.parse(file_path) 
            general_tag = xml_tree.getroot().find('General') 
            if general_tag is not None:
                writerID = int(general_tag[0].attrib.get("writerID", 0))
        except ElementTree.ParseError as e:
            print(f"Failed to parse XML file {file_path}: {e}")
        except ValueError as e:
            print(f"Failed to convert writerID to int in {file_path}")
    else:
        print(f"Warning: File not found or is not a file: {file_path}")
    return writerID

def get_text_line_by_line(file_path: Union[str, Path]) -> list[str]:
    """
    Reads a text file line by line.
    Returns a list of non-empty lines after the "CSR:" token if present. 
    """
    file_path = Path(file_path) 
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File {file_path} doesn't exist or is not a file") 
    
    lines = []
    with file_path.open('r') as f:
        csr_found = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            # start reading after CSR: is seen 
            if csr_found:
                lines.append(line)
            if line == "CSR:":
                csr_found = True
    # print(f"TEXT_LOADER: file {fname} with {len(lines)} text lines.")
    return lines

def get_stroke_seqs(stroke_file_path: Union[str, Path]) -> np.ndarray:
    """
    Parses an XML stroke file and returns the stroke points as a numpy array.
    Each point is represented as (x, y, eos) where eos indicates 
    the end of a continuous stroke (pen-up).
    """
    stroke_file_path = Path(stroke_file_path) 
    if not stroke_file_path.exists() or not stroke_file_path.is_file():
        raise FileNotFoundError(f"File {stroke_file_path} doesn't exist or is not a file!")
    
    xml_root = ElementTree.parse(stroke_file_path).getroot()
    stroke_set = xml_root.find('StrokeSet')
    if stroke_set is None:
        print(f"Warnning: No <StrokeSet> element found inside {stroke_file_path}")
        return np.empty((0,3), dtype=np.int32) 

    strokes = stroke_set.findall("Stroke")
    seqs = []
    for stroke in strokes:
        pts = len(stroke)
        for i, point in enumerate(stroke):
            if 'x' not in point.attrib or 'y' not in point.attrib:
                continue
            x = int(point.attrib['x'])
            y = -1 * int(point.attrib['y']) # negate y coord to match the whiteboard's coordinate convention
            # mark the last point in the stroke with eos flag = 1
            if i + 1 == pts:
                seqs.append((x, y, 1))
            else:
                seqs.append((x, y, 0))
    if not seqs:
        return np.empty((0,3), dtype=np.int32)

    return np.array(seqs, dtype=np.int32)

def list_files(root_dir: Union[str, Path], skip_hidden: bool = True) -> list[Path]:
    """
    Walks a directory and collects file paths.
    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        raise NotADirectoryError(f"Dirctory at {root_path} doesn't exist or it isn't direcotry")

    fnames = []
    # walk dirs and files recursively 
    for item in root_path.rglob('*'): 
        if item.is_file():
            if skip_hidden and item.name.startswith("."):
                continue 
            fnames.append(item) 
    return fnames