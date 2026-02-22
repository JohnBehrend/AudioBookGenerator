#!/usr/bin/env python3
"""
Module to parse chapter content and return chapter objects.
"""
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import argparse

class ChapterObj:
    def __init__(self, has_quotes: bool, text: str, line_num: int):
        self.has_quotes = has_quotes
        self.speaker = "narrator" # default to narrator
        self.text = text.strip()
        self.line_num = line_num
    def __str__(self):
        return f"Line {self.line_num}: Speaker {self.get_speaker()}: {self.text}"
    def get_speaker(self):
        return self.speaker
    def set_speaker(self, speaker):
        self.speaker = speaker

def get_chapter_objs(text: str):
    """
    Parse a chapter's text and return a list of chapter objects with the required properties.
    
    Args:
        text (str): The full text of the chapter
        
    Returns:
        list: List of ChapterObj objects with has_quotes, speaker, and text properties
    """
    # Split text into paragraphs (assuming paragraphs are separated by single newlines)
    paragraphs = text.split('\n')
    
    # Create a list of ChapterObj for each paragraph
    chapter_objs = []
    line_num = 0
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if paragraph.strip():
            # If there are quotes, we need to split the paragraph to separate the quote from the rest
            if '"' in paragraph:
                # Find all quoted text in the paragraph
                quotes = [x.strip() for x in paragraph.split('"')]
                if paragraph.startswith('"'):
                    quote_en=True
                else:
                    quote_en=False
                for quote in quotes:
                    if len(quote)>0:
                        line_num+=1
                        chapter_objs.append(ChapterObj(quote_en,  cleanup_text(quote), line_num))
                    quote_en = not quote_en
            else:
                # No quotes, treat as normal paragraph
                line_num+=1
                chapter_objs.append(ChapterObj(False, cleanup_text(paragraph), line_num))
    return chapter_objs

def parse_epub_to_chapters(epub_path, max_chapters=None):
    """
    Parse an EPUB file into an array of chapters.

    Args:
        epub_path (str): Path to the EPUB file
        max_chapters (int, optional): Maximum number of chapters to parse. If None, parses all chapters.

    Returns:
        list: Array of chapter texts
    """
    try:
        book = epub.read_epub(epub_path)
        chapters = []

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # Extract text content from HTML
                txt = BeautifulSoup(
                    item.get_content().decode('utf-8')
                    .replace(u"\xa0","")
                    .replace(u"\u2019","'")
                    .replace(u"\u201c",'\t"')
                    .replace(u"\u2014"," - ")
                    .replace(u"\u2026","...")
                    .replace(u"\u201d",'"')
                    .replace(u"\u2018","'")
                    .replace(u"\u2013","-")
                    # spaces
                    .replace(u"\u0009","") # character tabulation
                    .replace(u"\u000A","\n") # line feed
                    .replace(u"\u000B"," ") # line tabulation
                    .replace(u"\u000C"," ") # form feed
                    .replace(u"\u000D","\n") # carriage return
                    .replace(u"\u0020"," ") # space
                    .replace(u"\u0085","\n") # next line
                    .replace(u"\u00A0"," ") # no-break space
                    .replace(u"\u1680"," ") # ogham space mark
                    .replace(u"\u180E"," ") # mongolian vowel separator
                    .replace(u"\u2000"," ") # en quad
                    .replace(u"\u2001"," ") # em quad
                    .replace(u"\u2002"," ") # en space
                    .replace(u"\u2004"," ") # three-per-em space
                    .replace(u"\u2005"," ") # four-per-em space
                    .replace(u"\u2006"," ") # six-per-em space
                    .replace(u"\u2007"," ") # figure space
                    .replace(u"\u2008"," ") # punctuation space
                    .replace(u"\u2009"," ") # thin space
                    .replace(u"\u200A"," ") # hair space
                    .replace(u"\u200B"," ") # zero width space
                    .replace(u"\u200C"," ") # zero width non-joiner
                    .replace(u"\u200D"," ") # zero width joiner
                    .replace(u"\u2028","\n ") # line separator
                    .replace(u"\u2029","\n\n") # paragraph separator
                    .replace(u"\u202F"," ") # narrow no-break space
                    .replace(u"\u205F"," ") # medium mathematical space
                    .replace(u"\u2060","-") # word joiner
                    .replace(u"\u3000"," ") # ideographic space
                    , 'html.parser').get_text(separator=" ", strip=False)
                if txt.strip():  # Only add non-empty chapters
                    chapters.append(get_chapter_objs(txt))
                    # Stop if we've reached the max_chapters limit
                    if max_chapters is not None and len(chapters) >= max_chapters:
                        break
        return chapters

    except Exception as e:
        print(f"Error parsing EPUB file: {e}")
        return []

def cleanup_text(txt):
    txt = txt.replace("   ", " ")
    txt = txt.replace("  ", " ")
    return txt

def write_chapters_to_txt(chapters, output_dir, prefix="chapter_"):
    """Write chapter objects to text files.

    Args:
        chapters: List of chapter objects from parse_epub_to_chapters
        output_dir: Directory to write chapter files to
        prefix: Filename prefix for chapter files (default: "chapter_")

    Returns:
        List of paths to written chapter files
    """
    import os

    output_path = os.path.join(output_dir, prefix)
    written_files = []

    for i, chapter in enumerate(chapters):
        output_file = f"{output_path}{i}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for cobj in chapter:
                f.write(f"Line {cobj.line_num}: ")
                if cobj.has_quotes:
                    f.write('"')
                f.write(cobj.text)
                if cobj.has_quotes:
                    f.write('"')
                f.write("\n")
        written_files.append(output_file)

    return written_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an epub file into text files for each character. Quotes will always be split into unique lines.")
    parser.add_argument("-epub_file", help="Path to the EPUB file")
    args = parser.parse_args()
    chapters = parse_epub_to_chapters(args.epub_file)
    for i, chapter in enumerate(chapters):
        with open(f"./chapters/chapter_{i}.txt","w") as f:
            for cobj in chapter:
                f.write(f"Line {cobj.line_num}: ")
                if cobj.has_quotes:
                    f.write('"')
                f.write(cobj.text)
                if cobj.has_quotes:
                    f.write('"')
                f.write("\r\n")