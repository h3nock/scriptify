import os
import numpy as np
from xml.etree import ElementTree


def get_writerID(fname):
    """
    Reads a xml file and returns the writerID from general element (tag)
    """
    writerID = 0 
    if os.path.exists(fname):
        try:
            xml_tree = ElementTree.parse(fname) 
            general_tag = xml_tree.getroot().find('General') 
            if general_tag is not None:
                writerID = int(general_tag[0].attrib.get("writerID", 0))
        except ElementTree.ParseError as e:
            print(f"Warning: Failed to parse XML file {fname}: {e}")
    return writerID

def get_text_line_by_line(fname):
    """
    Reads a text file line by line.
    Returns a list of non-empty lines after the "CSR:" token if present. 
    """
    if not os.path.exists(fname):
        raise ValueError(f"File {fname} doesn't exist!")
    
    lines = []
    with open(fname, 'r') as f:
        csr_found = False
        for line in f.readlines():
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

def get_stroke_seqs(stroke_file):
    """
    Parses an XML stroke file and returns the stroke points as a numpy array.
    Each point is represented as (x, y, eos) where eos indicates 
    the end of a continuous stroke (pen-up).
    """
    if not os.path.exists(stroke_file):
        raise ValueError(f"File {stroke_file} doesn't exist!")
    
    xml_root = ElementTree.parse(stroke_file).getroot()
    stroke_set = xml_root.find('StrokeSet')
    if stroke_set is None:
        raise ValueError(f"No <StrokeSet> element found inside {stroke_file}")
    
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

def list_files(root_dir, skip_hidden=True):
    """
    Walks a directory and collects file paths.
    """
    fnames = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if skip_hidden and filename.startswith('.'):
                continue
            fnames.append(os.path.join(dirpath, filename))
    return fnames