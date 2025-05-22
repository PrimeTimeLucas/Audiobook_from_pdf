### Parameters

- **`pdf_path`**: Path to your PDF file
- **`output_dir`**: Directory where audio files will be saved (optional)
- **`chunk_size`**: Number of PDF pages per audio chunk (default: 10)
- **`lang`**: Language code (default: 'en')
- **`tld`**: Top-level domain for accent (default: 'co.za')
- **`slow`**: Use slower speech (# Enhanced PDF to Audiobook Converter

Convert PDF books into high-quality audiobooks using Google Text-to-Speech (gTTS) with advanced preprocessing and tokenization features for natural-sounding speech.

## 🎯 Features

- **High-Quality Speech**: Uses Google TTS with South African accent support
- **Faster Playback**: Configurable speed adjustment (default 1.3x faster)
- **Smart Content Detection**: Automatically skips title pages and front matter
- **Advanced Preprocessing**: Handles academic texts, mathematical notation, citations
- **Smart Tokenization**: Natural speech flow with proper pauses and intonation
- **Chunked Processing**: Processes large PDFs in manageable chunks
- **Enhanced Pronunciation**: Custom word substitutions for better speech quality
- **Comprehensive Logging**: Detailed conversion logs and summaries
- **Playlist Generation**: Automatic M3U playlist creation for easy playback

## 📋 Requirements

- Python 3.7 or higher
- Internet connection (for Google TTS API)

## 🚀 Installation

1. **Clone or download the script files:**
   - `enhanced_pdf_audiobook.py` (main script)
   - `requirements.txt`

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Alternative manual installation:**
   ```bash
   pip install PyPDF2>=3.0.0 gTTS>=2.3.0 pydub>=0.25.0
   ```

## 📖 Usage

### Basic Usage

1. **Place your PDF file in the same directory as the script**
2. **Update the PDF filename in the script:**
   ```python
   pdf_path = 'your_book_name.pdf'  # Change this to your PDF filename
   ```
3. **Run the script:**
   ```bash
   python enhanced_pdf_audiobook.py
   ```

### Command Line Usage

```bash
# Basic conversion
python enhanced_pdf_audiobook.py "your_book.pdf"

# With custom output directory
python enhanced_pdf_audiobook.py "your_book.pdf" "my_audiobook"

# With custom chunk size (pages per audio file)
python enhanced_pdf_audiobook.py "your_book.pdf" "my_audiobook" 15
```

### Programmatic Usage

```python
from enhanced_pdf_audiobook import AdvancedPDFToAudiobookGTTS

# Basic conversion
converter = AdvancedPDFToAudiobookGTTS('your_book.pdf')
successful_chunks = converter.convert()

# Advanced configuration
converter = AdvancedPDFToAudiobookGTTS(
    pdf_path='academic_paper.pdf',
    output_dir='my_audiobook',
    chunk_size=15,             # 15 pages per audio chunk
    lang='en',                 # Language
    tld='co.za',              # South African accent
    slow=False,               # Normal speech speed
    speed_factor=1.3,         # 30% faster playback
    skip_to_content=True,     # Skip front matter automatically
    start_keywords=['chapter 1', 'introduction']  # Custom content start keywords
)
converter.convert()
```

## ⚙️ Configuration Options

### Language and Accent Options

| TLD | Accent |
|-----|--------|
| `'co.za'` | South African English |
| `'com.au'` | Australian English |
| `'co.uk'` | British English |
| `'com'` | American English |
| `'ca'` | Canadian English |

### Parameters

- **`pdf_path`**: Path to your PDF file
- **`output_dir`**: Directory where audio files will be saved (optional)
- **`chunk_size`**: Number of PDF pages per audio chunk (default: 10)
- **`lang`**: Language code (default: 'en')
- **`tld`**: Top-level domain for accent (default: 'co.za')
- **`slow`**: Use slower speech (default: False)

## 📁 Output Structure

After conversion, you'll get:

```
your_book_audiobook_gtts/
├── chunk_001.mp3
├── chunk_002.mp3
├── chunk_003.mp3
├── ...
├── audiobook_playlist.m3u
├── conversion.log
└── conversion_summary.txt
```

### File Descriptions

- **`chunk_XXX.mp3`**: Audio chunks (numbered sequentially)
- **`audiobook_playlist.m3u`**: Playlist file for easy playback
- **`conversion.log`**: Detailed conversion log
- **`conversion_summary.txt`**: Summary of conversion results

## 🎵 Playing Your Audiobook

### Using the Playlist
1. Open `audiobook_playlist.m3u` in your preferred media player
2. Supported players: VLC, Windows Media Player, iTunes, etc.

### Manual Playback
Play the chunks in order: `chunk_001.mp3`, `chunk_002.mp3`, etc.

## 🔧 Advanced Features

### Enhanced Preprocessing

The converter includes advanced preprocessing for:

- **Academic Texts**: Handles citations, references, equations
- **Mathematical Notation**: Converts symbols to spoken words (π → "pi")
- **Abbreviations**: Smart handling of common abbreviations
- **Technical Terms**: Proper pronunciation of scientific terms
- **URLs and Emails**: Replaces with "web link" and "email address"

### Custom Word Substitutions

Automatically handles:
- `Ph.D.` → "PhD"
- `et al.` → "et alia"
- `i.e.` → "that is"
- `e.g.` → "for example"
- Mathematical symbols and units

## 🛠️ Troubleshooting

### Common Issues

1. **"No text to speak" Error**
   - Check if your PDF contains readable text (not just images)
   - Try a different PDF or scan quality

2. **Network Errors**
   - Ensure stable internet connection
   - Check if Google services are accessible

3. **Very Large Files**
   - Reduce `chunk_size` parameter
   - Process in smaller batches

4. **Audio Quality Issues**
   - Try `slow=True` for clearer speech
   - Experiment with different `tld` values for accents

### File Size Considerations

- Each chunk typically produces 1-5 MB audio files
- Larger chunk sizes = fewer files but longer processing time
- Recommended chunk size: 10-15 pages

## 📝 Tips for Best Results

1. **PDF Quality**: Use high-quality PDFs with clear text
2. **Chunk Size**: Balance between file count and processing time
3. **Academic Content**: The converter is optimized for academic/technical texts
4. **Internet**: Ensure stable connection during conversion
5. **File Names**: Avoid special characters in PDF filenames

## 🆘 Support

### Check Logs
Always check the `conversion.log` file for detailed error information.

### Common Solutions
- Update dependencies: `pip install --upgrade PyPDF2 gTTS`
- Check Python version: Requires Python 3.7+
- Verify PDF integrity: Try opening in a PDF reader first

## 📄 License

This project uses the MIT License. Feel free to modify and distribute.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional language support
- Voice customization options
- Batch processing features
- GUI interface

---

**Happy listening! 🎧**