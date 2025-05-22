import PyPDF2
from gtts import gTTS
from gtts.tokenizer import pre_processors, tokenizer_cases
from gtts.tokenizer.core import PreProcessorSub, PreProcessorRegex, Tokenizer
import gtts.tokenizer.symbols
import os
import re
import time
import sys
from pathlib import Path
from typing import Optional, List
import logging

class AdvancedPDFToAudiobookGTTS:
    def __init__(self, 
                 pdf_path: str,
                 output_dir: str = None,
                 chunk_size: int = 10,
                 lang: str = 'en',
                 tld: str = 'co.za',
                 slow: bool = False,
                 speed_factor: float = 1.3,
                 skip_to_content: bool = True,
                 start_keywords: List[str] = None):
        """
        Advanced PDF to Audiobook converter using Google TTS with enhanced preprocessing
        
        Args:
            pdf_path (str): Path to the PDF file
            output_dir (str): Directory to save audio chunks (optional)
            chunk_size (int): Number of pages per audio chunk
            lang (str): Language code (default: 'en')
            tld (str): Top-level domain for accent ('co.za' for South African)
            slow (bool): Whether to use slow speech
            speed_factor (float): Speed multiplier for audio playback (1.0 = normal, 1.3 = 30% faster)
            skip_to_content (bool): Whether to automatically skip to main content
            start_keywords (List[str]): Custom keywords to identify content start (optional)
        """
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.lang = lang
        self.tld = tld
        self.slow = slow
        self.speed_factor = speed_factor
        self.skip_to_content = skip_to_content
        
        # Default keywords to identify content start
        self.start_keywords = start_keywords or [
            'preface', 'foreword', 'introduction', 'chapter 1', 'chapter one',
            'prologue', 'overview', 'abstract', 'executive summary',
            'table of contents', 'contents'
        ]
        
        # Set up output directory
        if output_dir is None:
            pdf_name = Path(pdf_path).stem
            self.output_dir = f"{pdf_name}_audiobook_gtts"
        else:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging
        self.setup_logging()
        
        # Initialize custom preprocessors and tokenizer
        self.setup_custom_preprocessing()
        
        self.logger.info(f"Initialized advanced converter for: {pdf_path}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Language: {lang}, TLD: {tld}, Slow: {slow}")
        self.logger.info(f"Speed factor: {speed_factor}x")
        self.logger.info(f"Skip to content: {skip_to_content}")
        if skip_to_content:
            self.logger.info(f"Start keywords: {', '.join(self.start_keywords)}")
    
    def setup_logging(self):
        """Set up logging configuration"""
        log_file = os.path.join(self.output_dir, 'conversion.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_custom_preprocessing(self):
        """Set up custom preprocessing functions for better speech quality"""
        
        # Add custom word substitutions for better pronunciation
        custom_substitutions = [
            # Common academic/scientific terms
            ('Ph.D.', 'PhD'),
            ('et al.', 'et alia'),
            ('i.e.', 'that is'),
            ('e.g.', 'for example'),
            ('vs.', 'versus'),
            ('etc.', 'et cetera'),
            ('cf.', 'compare'),
            ('viz.', 'namely'),
            ('w.r.t.', 'with respect to'),
            
            # Numbers and measurements
            ('kg', 'kilograms'),
            ('km', 'kilometers'),
            ('cm', 'centimeters'),
            ('mm', 'millimeters'),
            ('°C', 'degrees Celsius'),
            ('°F', 'degrees Fahrenheit'),
            ('%', ' percent'),
            
            # Common abbreviations in academic texts
            ('Fig.', 'Figure'),
            ('Eq.', 'Equation'),
            ('Ref.', 'Reference'),
            ('Ch.', 'Chapter'),
            ('Sec.', 'Section'),
            ('Vol.', 'Volume'),
            ('No.', 'Number'),
            ('p.', 'page'),
            ('pp.', 'pages'),
            
            # Mathematical terms
            ('∞', 'infinity'),
            ('α', 'alpha'),
            ('β', 'beta'),
            ('γ', 'gamma'),
            ('δ', 'delta'),
            ('π', 'pi'),
            ('σ', 'sigma'),
            ('μ', 'mu'),
            ('λ', 'lambda'),
            
            # Common contractions that might not be handled well
            ("won't", "will not"),
            ("can't", "cannot"),
            ("shouldn't", "should not"),
            ("wouldn't", "would not"),
            ("couldn't", "could not"),
            ("mustn't", "must not"),
        ]
        
        # Add custom substitutions to the global list
        for old, new in custom_substitutions:
            if (old, new) not in gtts.tokenizer.symbols.SUB_PAIRS:
                gtts.tokenizer.symbols.SUB_PAIRS.append((old, new))
        
        # Add custom abbreviations that should have periods removed
        custom_abbreviations = [
            'PhD', 'MSc', 'BSc', 'MD', 'DDS', 'JD', 'LLB', 'MBA',
            'CEO', 'CTO', 'CFO', 'COO', 'VP', 'SVP', 'EVP',
            'USA', 'UK', 'EU', 'UN', 'WHO', 'NASA', 'IBM', 'AI',
            'ML', 'DL', 'NLP', 'CV', 'GPU', 'CPU', 'RAM', 'SSD',
            'HTML', 'CSS', 'JS', 'SQL', 'API', 'UI', 'UX', 'QA'
        ]
        
        for abbrev in custom_abbreviations:
            if abbrev not in gtts.tokenizer.symbols.ABBREVIATIONS:
                gtts.tokenizer.symbols.ABBREVIATIONS.append(abbrev)
        
        self.logger.info(f"Added {len(custom_substitutions)} custom word substitutions")
        self.logger.info(f"Added {len(custom_abbreviations)} custom abbreviations")
    
    def find_content_start_page(self, reader: PyPDF2.PdfReader) -> int:
        """
        Find the page where main content starts by looking for keywords
        
        Args:
            reader: PyPDF2 reader object
            
        Returns:
            int: Page index where content starts (0-based)
        """
        if not self.skip_to_content:
            return 0
        
        total_pages = len(reader.pages)
        max_search_pages = min(50, total_pages)  # Don't search beyond first 50 pages
        
        self.logger.info("Searching for content start page...")
        
        for page_idx in range(max_search_pages):
            try:
                page = reader.pages[page_idx]
                page_text = page.extract_text().lower()
                
                # Check for start keywords
                for keyword in self.start_keywords:
                    if keyword.lower() in page_text:
                        self.logger.info(f"Found '{keyword}' on page {page_idx + 1}, starting from here")
                        return page_idx
                
                # Additional heuristics
                # Look for "Chapter" followed by number or "1"
                if re.search(r'chapter\s+(?:1|one|i)\b', page_text, re.IGNORECASE):
                    self.logger.info(f"Found 'Chapter 1' pattern on page {page_idx + 1}")
                    return page_idx
                
                # Look for significant text content (not just title pages)
                words = page_text.split()
                if len(words) > 100:  # Page has substantial content
                    # Check if it's likely content vs. front matter
                    front_matter_indicators = [
                        'copyright', 'isbn', 'published', 'publisher', 'edition',
                        'all rights reserved', 'library of congress'
                    ]
                    
                    has_front_matter = any(indicator in page_text for indicator in front_matter_indicators)
                    
                    if not has_front_matter and page_idx > 5:  # Not front matter and past title pages
                        self.logger.info(f"Found substantial content on page {page_idx + 1}")
                        return page_idx
                        
            except Exception as e:
                self.logger.debug(f"Error reading page {page_idx + 1}: {e}")
                continue
        
        # If no content start found, start from page 1 (skip only title page)
        self.logger.info("No specific content start found, starting from page 2")
        return 1 if total_pages > 1 else 0
    
    def create_custom_preprocessor(self):
        """Create a custom preprocessor function for academic texts"""
        def academic_text_preprocessor(text):
            """Custom preprocessor for academic and technical texts"""
            
            # Fix common PDF extraction issues
            text = re.sub(r'([a-z])([A-Z])', r'\1. \2', text)  # Add periods between sentences
            text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Space after punctuation
            
            # Handle citations like [1], [Smith, 2020], etc.
            text = re.sub(r'\[([^\]]+)\]', r'reference \1', text)
            
            # Handle equations and formulas
            text = re.sub(r'\$([^$]+)\$', r'equation \1', text)  # LaTeX inline math
            text = re.sub(r'\\([a-zA-Z]+)', r'\1', text)  # Remove LaTeX commands
            
            # Handle footnotes and superscripts
            text = re.sub(r'\^(\d+)', r'superscript \1', text)
            text = re.sub(r'_(\d+)', r'subscript \1', text)
            
            # Handle URLs
            text = re.sub(r'https?://[^\s]+', 'web link', text)
            text = re.sub(r'www\.[^\s]+', 'web link', text)
            
            # Handle email addresses
            text = re.sub(r'\S+@\S+\.\S+', 'email address', text)
            
            # Handle file paths
            text = re.sub(r'[A-Z]:\\[^\s]+', 'file path', text)
            text = re.sub(r'/[^\s]*/', 'file path', text)
            
            # Improve number reading
            text = re.sub(r'(\d+)\.(\d+)', r'\1 point \2', text)  # Decimals
            text = re.sub(r'(\d{4})-(\d{4})', r'\1 to \2', text)  # Year ranges
            text = re.sub(r'(\d+)-(\d+)', r'\1 to \2', text)  # Number ranges
            
            # Handle parenthetical expressions better
            text = re.sub(r'\(([^)]+)\)', r', \1,', text)
            
            return text
        
    def apply_speed_adjustment(self, filename: str) -> bool:
        """
        Apply speed adjustment to the audio file using audio processing
        
        Args:
            filename (str): Path to the audio file
            
        Returns:
            bool: Success status
        """
        if self.speed_factor == 1.0:
            return True  # No speed adjustment needed
        
        try:
            # Try using pydub if available for better quality
            try:
                from pydub import AudioSegment
                from pydub.effects import speedup
                
                # Load audio
                audio = AudioSegment.from_mp3(filename)
                
                # Apply speed adjustment
                if self.speed_factor != 1.0:
                    # Use speedup for better quality than just changing frame rate
                    adjusted_audio = speedup(audio, playback_speed=self.speed_factor)
                else:
                    adjusted_audio = audio
                
                # Save adjusted audio
                adjusted_audio.export(filename, format="mp3")
                self.logger.debug(f"Applied {self.speed_factor}x speed using pydub")
                return True
                
            except ImportError:
                # Fallback: Create an instruction file for manual speed adjustment
                speed_info_file = filename.replace('.mp3', '_speed_info.txt')
                with open(speed_info_file, 'w') as f:
                    f.write(f"Recommended playback speed: {self.speed_factor}x\n")
                    f.write("To adjust speed:\n")
                    f.write("- VLC: Playback > Speed > Faster/Slower\n")
                    f.write("- Most media players have speed controls\n")
                    f.write(f"- Set speed to {self.speed_factor}x for optimal listening\n")
                
                self.logger.info(f"pydub not available. Created speed info file: {speed_info_file}")
                return True
                
        except Exception as e:
            self.logger.warning(f"Speed adjustment failed: {e}")
            return True  # Don't fail the conversion for speed issues
    
    def get_enhanced_preprocessors(self):
        """Get the list of preprocessors including custom ones"""
        try:
            # Try to use all enhanced preprocessors
            return [
                pre_processors.tone_marks,          # Handle tone marks
                pre_processors.end_of_line,         # Fix hyphenated words
                self.create_custom_preprocessor(),  # Our custom academic preprocessor
                pre_processors.abbreviations,       # Remove periods from abbreviations
                pre_processors.word_sub,           # Word substitutions
            ]
        except Exception as e:
            self.logger.warning(f"Could not load all preprocessors: {e}")
            # Fallback to basic preprocessors
            try:
                return [
                    self.create_custom_preprocessor(),  # Our custom preprocessor should always work
                    pre_processors.abbreviations,       # Basic abbreviation handling
                    pre_processors.word_sub,           # Basic word substitution
                ]
            except Exception as e2:
                self.logger.warning(f"Using minimal preprocessing due to errors: {e2}")
                # Minimal fallback
                return [self.create_custom_preprocessor()]
    
    def get_enhanced_tokenizer(self):
        """Get enhanced tokenizer for better speech flow"""
        try:
            # Try to create the tokenizer with the standard cases
            tokenizer = Tokenizer([
                tokenizer_cases.tone_marks,
                tokenizer_cases.period_comma,
                tokenizer_cases.colon,
                tokenizer_cases.other_punctuation
            ])
            return tokenizer.run
        except Exception as e:
            self.logger.warning(f"Could not create custom tokenizer: {e}")
            # Return None to use gTTS default tokenizer
            return None
    
    def clean_text(self, text: str) -> str:
        """
        Enhanced text cleaning with better preprocessing
        
        Args:
            text (str): Raw text from PDF
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Basic cleaning
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        lines = text.split('.')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines, standalone numbers, or common headers
            if (len(line) > 10 and 
                not line.isdigit() and 
                not re.match(r'^(page|chapter|\d+)$', line.lower()) and
                not re.match(r'^[ivxlc]+$', line.lower())):  # Roman numerals
                cleaned_lines.append(line)
        
        cleaned_text = '. '.join(cleaned_lines)
        
        # Apply the custom preprocessors manually for preview
        custom_preprocessor = self.create_custom_preprocessor()
        cleaned_text = custom_preprocessor(cleaned_text)
        
        return cleaned_text.strip()
    
    def extract_text_from_pages(self, reader: PyPDF2.PdfReader, start: int, end: int) -> str:
        """
        Extract and clean text from a range of pages
        
        Args:
            reader: PyPDF2 reader object
            start (int): Starting page index
            end (int): Ending page index
            
        Returns:
            str: Extracted and cleaned text
        """
        text = ''
        
        for i in range(start, end):
            try:
                page = reader.pages[i]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
                self.logger.debug(f"Extracted text from page {i + 1}")
            except Exception as e:
                self.logger.warning(f"Error extracting text from page {i + 1}: {e}")
                continue
        
        return self.clean_text(text)
    
    def create_audio_chunk(self, text: str, chunk_number: int, start_page: int, end_page: int) -> bool:
        """
        Create an audio file from text using Google TTS with enhanced preprocessing
        
        Args:
            text (str): Text to convert to speech
            chunk_number (int): Chunk number for filename
            start_page (int): Starting page number
            end_page (int): Ending page number
            
        Returns:
            bool: Success status
        """
        if not text.strip():
            self.logger.warning(f"No text found for chunk {chunk_number}")
            return False
        
        # Check text length
        if len(text) > 5000:
            self.logger.warning(f"Chunk {chunk_number} is very long ({len(text)} chars). Consider smaller chunk size.")
        
        try:
            filename = os.path.join(self.output_dir, f'chunk_{chunk_number:03d}.mp3')
            
            # Create gTTS with enhanced preprocessing and tokenizing
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Get the tokenizer function
                    tokenizer_func = self.get_enhanced_tokenizer()
                    preprocessors = self.get_enhanced_preprocessors()
                    
                    # Create gTTS instance with custom settings
                    if tokenizer_func and preprocessors:
                        tts = gTTS(
                            text=text,
                            lang=self.lang,
                            tld=self.tld,
                            slow=self.slow,
                            pre_processor_funcs=preprocessors,
                            tokenizer_func=tokenizer_func
                        )
                    elif preprocessors:
                        # Use preprocessors without custom tokenizer
                        tts = gTTS(
                            text=text,
                            lang=self.lang,
                            tld=self.tld,
                            slow=self.slow,
                            pre_processor_funcs=preprocessors
                        )
                    else:
                        # Fallback to basic gTTS
                        self.logger.warning("Using basic gTTS without custom preprocessing")
                        tts = gTTS(
                            text=text,
                            lang=self.lang,
                            tld=self.tld,
                            slow=self.slow
                        )
                    
                    tts.save(filename)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Attempt {attempt + 1} failed for chunk {chunk_number}: {e}. Retrying...")
                        time.sleep(2)
                    else:
                        raise e
            
            # Verify file was created
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                # Apply speed adjustment if needed
                self.apply_speed_adjustment(filename)
                
                file_size = os.path.getsize(filename) / 1024  # KB
                speed_info = f" at {self.speed_factor}x speed" if self.speed_factor != 1.0 else ""
                self.logger.info(f"✓ Chunk {chunk_number:03d}: pages {start_page}-{end_page} "
                               f"({len(text)} chars, {file_size:.1f} KB{speed_info})")
                return True
            else:
                self.logger.error(f"Failed to create valid audio file for chunk {chunk_number}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error creating audio for chunk {chunk_number}: {e}")
            return False
    
    def convert(self) -> List[str]:
        """
        Main conversion method with enhanced preprocessing
        
        Returns:
            List[str]: List of successfully created audio files
        """
        if not os.path.exists(self.pdf_path):
            self.logger.error(f"PDF file not found: {self.pdf_path}")
            return []
        
        self.logger.info("Starting enhanced PDF to audiobook conversion...")
        self.logger.info("Using advanced preprocessing and tokenization features")
        
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                # Find content start page
                start_page_idx = self.find_content_start_page(reader)
                pages_to_process = total_pages - start_page_idx
                
                self.logger.info(f"PDF has {total_pages} pages total")
                self.logger.info(f"Starting from page {start_page_idx + 1}, processing {pages_to_process} pages")
                self.logger.info(f"Creating chunks of {self.chunk_size} pages each")
                
                successful_chunks = []
                chunk_number = 1
                
                for start in range(start_page_idx, total_pages, self.chunk_size):
                    end = min(start + self.chunk_size, total_pages)
                    
                    self.logger.info(f"Processing chunk {chunk_number}: pages {start + 1} to {end}")
                    
                    # Extract text from pages
                    text = self.extract_text_from_pages(reader, start, end)
                    
                    if text.strip():
                        success = self.create_audio_chunk(text, chunk_number, start + 1, end)
                        if success:
                            successful_chunks.append(f'chunk_{chunk_number:03d}.mp3')
                        
                        # Add delay to avoid rate limits
                        time.sleep(0.5)
                    else:
                        self.logger.warning(f"No readable text found in chunk {chunk_number}")
                    
                    chunk_number += 1
                
                self.create_summary(successful_chunks, total_pages, start_page_idx)
                return successful_chunks
                
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            return []
    
    def create_summary(self, successful_chunks: List[str], total_pages: int, start_page: int):
        """Create a summary file with conversion details"""
        summary_file = os.path.join(self.output_dir, 'conversion_summary.txt')
        
        with open(summary_file, 'w') as f:
            f.write("Enhanced PDF to Audiobook Conversion Summary\n")
            f.write("=" * 45 + "\n\n")
            f.write(f"Source PDF: {self.pdf_path}\n")
            f.write(f"Total pages in PDF: {total_pages}\n")
            f.write(f"Started from page: {start_page + 1}\n")
            f.write(f"Pages processed: {total_pages - start_page}\n")
            f.write(f"Chunk size: {self.chunk_size} pages\n")
            f.write(f"Language: {self.lang}\n")
            f.write(f"Accent (TLD): {self.tld}\n")
            f.write(f"Slow speech: {self.slow}\n")
            f.write(f"Speed factor: {self.speed_factor}x\n")
            f.write(f"Successful chunks: {len(successful_chunks)}\n\n")
            
            f.write("Enhanced Features Used:\n")
            f.write("  ✓ Custom academic text preprocessing\n")
            f.write("  ✓ Enhanced abbreviation handling\n")
            f.write("  ✓ Improved word substitutions\n")
            f.write("  ✓ Better tokenization for natural speech\n")
            f.write("  ✓ Mathematical notation handling\n")
            f.write("  ✓ Citation and reference processing\n")
            if self.skip_to_content:
                f.write("  ✓ Automatic content detection (skipped front matter)\n")
            if self.speed_factor != 1.0:
                f.write(f"  ✓ Speed adjustment ({self.speed_factor}x)\n")
            f.write("\n")
            
            f.write("Audio files created:\n")
            for chunk in successful_chunks:
                f.write(f"  - {chunk}\n")
            
            f.write(f"\nAll files saved in: {self.output_dir}\n")
            
            if self.speed_factor != 1.0:
                f.write(f"\nPlayback Notes:\n")
                f.write(f"- Audio optimized for {self.speed_factor}x playback speed\n")
                f.write(f"- If speed adjustment wasn't applied automatically,\n")
                f.write(f"  set your media player to {self.speed_factor}x speed\n")
        
        self.logger.info(f"Conversion summary saved: {summary_file}")

def create_playlist(output_dir: str, chunks: List[str]):
    """Create an M3U playlist file for easy playbook"""
    playlist_file = os.path.join(output_dir, 'audiobook_playlist.m3u')
    
    with open(playlist_file, 'w') as f:
        f.write("#EXTM3U\n")
        f.write("#EXTINF:-1,Enhanced PDF Audiobook\n")
        for chunk in sorted(chunks):
            f.write(f"{chunk}\n")
    
    print(f"📝 Playlist created: {playlist_file}")

def main():
    """Main execution with enhanced features"""
    
    # Configuration - Update this path to your PDF
    pdf_path = 'The book of why_ the new science of cause and effect ( PDFDrive ).pdf'
    
    # Create enhanced converter
    converter = AdvancedPDFToAudiobookGTTS(
        pdf_path=pdf_path,
        output_dir='enhanced_audiobook_chunks_gtts',
        chunk_size=5,      # pages per chunk
        lang='en',
        tld='co.za',        # South African accent
        slow=False,
        speed_factor=1.3,   # 30% faster playback
        skip_to_content=True  # Skip front matter
    )
    
    # Convert PDF to audiobook
    print("🎙️ Starting enhanced PDF to audiobook conversion...")
    print("Using advanced gTTS preprocessing and tokenization features")
    
    successful_chunks = converter.convert()
    
    print(f"\n🎉 Enhanced conversion complete!")
    print(f"Created {len(successful_chunks)} audio chunks with improved speech quality")
    print(f"Files saved in: {converter.output_dir}")
    
    # Create playlist
    if successful_chunks:
        create_playlist(converter.output_dir, successful_chunks)
    
    print("\n📚 Enhanced features used:")
    print("  ✓ Academic text preprocessing")
    print("  ✓ Better abbreviation handling")
    print("  ✓ Improved pronunciation corrections")
    print("  ✓ Natural speech tokenization")
    print("  ✓ Mathematical notation support")
    print("  ✓ Citation and reference processing")

if __name__ == "__main__":
    # Command line usage
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        output_directory = sys.argv[2] if len(sys.argv) > 2 else None
        chunk_pages = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        
        converter = AdvancedPDFToAudiobookGTTS(
            pdf_path=pdf_file,
            output_dir=output_directory,
            chunk_size=chunk_pages,
            lang='en',
            tld='co.za',
            speed_factor=1.3,
            skip_to_content=True
        )
        converter.convert()
    else:
        main()