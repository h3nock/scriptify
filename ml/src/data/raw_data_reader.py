import numpy as np
from pathlib import Path
from typing import NamedTuple, Union 
from xml.etree import ElementTree

class SourceDocumentInfo(NamedTuple):
    doc_id: str 
    text_lines: list[str] 
    line_stroke_file_paths: list[Path] 
    writer_id: int 
    original_text_file_path: Path
    original_xml_path: Path
    
DEFAULT_ASCII_SUBDIR = 'ascii' 
DEFAULT_STROKES_SUBDIR = 'lineStrokes' 
DEFAULT_ORIGINAL_XML_SUBDIR = 'original' 

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

def collect_and_organize_docs(
    data_root_path: Union['str', Path] = "./data/raw/", 
    ascii_subdir_name: str = DEFAULT_ASCII_SUBDIR, 
    strokes_subdir_name: str = DEFAULT_STROKES_SUBDIR, 
    xml_subdir_name: str = DEFAULT_ORIGINAL_XML_SUBDIR
) -> list[SourceDocumentInfo]:

    collected_docs: list[SourceDocumentInfo] = [] 
    data_root_path = Path(data_root_path)
    stroke_file_counts = 0
    print(f"data_root_path: {data_root_path.resolve()}") 
    if not data_root_path.exists():
        raise FileNotFoundError(f"Root data folder doesn't exis at {data_root_path}")

    ascii_root_path = data_root_path / ascii_subdir_name 
    raw_line_strokes_dir = data_root_path / strokes_subdir_name
    raw_original_xml_dir = data_root_path / xml_subdir_name 
    
    if not ascii_root_path.is_dir():
        raise FileNotFoundError(f"ASCII directory not found at {ascii_root_path}")
    
    if not raw_line_strokes_dir.is_dir():
        raise FileNotFoundError(f"Strokes directory not found at {raw_line_strokes_dir}")
    
    
    if not raw_original_xml_dir.is_dir():
        raise FileNotFoundError(f"Original xml directory not found at {raw_original_xml_dir}")
    
    all_ascii_files = list_files(ascii_root_path) 
    for ascii_file_path in all_ascii_files:
        # sample ascii_file_path: ascii/a01/a01-000/a01-000u.txt 
        try:
            doc_id_stem = ascii_file_path.stem 
            doc_parent_dir_name = ascii_file_path.parent.name 
            text_lines = get_text_line_by_line(ascii_file_path)

            if not text_lines:
                continue 
            
            # relative_dir = a01/a01-000/, given ascii_file_path = ascii/a01/a01-000/a01-000u.txt  
            relative_dir = ascii_file_path.parent.relative_to(ascii_root_path)

            # stroke_file_base: .../lineStrokes/a01/a01-000 
            stroke_files_dir_for_doc = raw_line_strokes_dir / relative_dir 

            # orginal_xml_base_dir 
            original_xml_dir_for_doc = raw_original_xml_dir / relative_dir
            
            file_variant_suffix = doc_id_stem[-1]
            if not file_variant_suffix.isalpha():
                file_variant_suffix = ""
            
            if not stroke_files_dir_for_doc.is_dir():
                continue

            # stroke_file_name_prefix: a01-000u-
            stroke_file_name_prefix = f"{doc_parent_dir_name}{file_variant_suffix}-"
            original_xml_filename = f"strokes{file_variant_suffix}.xml"
            original_xml_path = original_xml_dir_for_doc / original_xml_filename

            if not original_xml_path.is_file():
                continue
            
            # list of filenames where each file corresponds to stroke respresentation of a single text line 
            line_stroke_files: list[Path] = sorted(
                stroke_files_dir_for_doc.glob(f"{stroke_file_name_prefix}*.xml"),
                key=lambda p: int(p.stem.split('-')[-1])
            )

            if len(line_stroke_files) != len(text_lines):
                continue

            writer_id = get_writerID(original_xml_path)

            stroke_file_counts += len(line_stroke_files)
            collected_docs.append(
                SourceDocumentInfo(
                    doc_id=doc_id_stem,
                    text_lines=text_lines,
                    line_stroke_file_paths=line_stroke_files,
                    writer_id=writer_id,
                    original_text_file_path=ascii_file_path,
                    original_xml_path=original_xml_path,
                )
            )

        except FileNotFoundError as e:
            print(f"Skipping {ascii_file_path.name} due to missing file during collection: {e}")
        except ValueError as e: 
            print(f"Skipping {ascii_file_path.name} due to value error: {e}")
        except Exception as e:
            print(f"Unexpected error collating document for {ascii_file_path.name}: {e}")
    print(f"{len(collected_docs)} documents collected!")
    print(f"Total number of stroke files: {stroke_file_counts}")
    return collected_docs 

if __name__ == "__main__":
    collect_and_organize_docs(data_root_path='./data/raw/')
