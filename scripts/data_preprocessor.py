# scripts/data_preprocessor.py - Fixed version with proper NLTK downloads
import pandas as pd
import numpy as np
import re
import string
import logging

# Import NLTK with proper downloads
try:
    import nltk
    
    # Download required NLTK data with error handling
    def download_nltk_data():
        """Download required NLTK data with proper error handling"""
        required_data = [
            ('punkt', 'tokenizers/punkt'),
            ('punkt_tab', 'tokenizers/punkt_tab'),
            ('stopwords', 'corpora/stopwords')
        ]
        
        for name, path in required_data:
            try:
                nltk.data.find(path)
                print(f"NLTK {name} data already exists")
            except LookupError:
                print(f"Downloading NLTK {name} data...")
                try:
                    nltk.download(name, quiet=True)
                    print(f"Successfully downloaded {name}")
                except Exception as e:
                    print(f"Failed to download {name}: {e}")
                    # Continue execution - we'll handle missing data later
    
    # Download required data
    download_nltk_data()
    
    # Import tokenization functions
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
    # Test if downloads worked
    try:
        test_tokens = word_tokenize("test sentence")
        indonesian_stopwords = set(stopwords.words('indonesian'))
        NLTK_AVAILABLE = True
        print("NLTK setup successful")
    except Exception as e:
        print(f"NLTK test failed: {e}")
        NLTK_AVAILABLE = False

except ImportError:
    print("NLTK not available")
    NLTK_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewPreprocessor:
    def __init__(self):
        # Setup stopwords
        if NLTK_AVAILABLE:
            try:
                # Indonesian stopwords from NLTK
                self.indonesian_stopwords = set(stopwords.words('indonesian'))
            except Exception as e:
                logger.warning(f"Could not load Indonesian stopwords from NLTK: {e}")
                # Fallback to manual list
                self.indonesian_stopwords = self.get_manual_indonesian_stopwords()
        else:
            # Manual Indonesian stopwords
            self.indonesian_stopwords = self.get_manual_indonesian_stopwords()
        
        # Additional common Indonesian stopwords for app reviews
        additional_stopwords = {
            'app', 'aplikasi', 'bagus', 'jelek', 'baik', 'buruk',
            'suka', 'tidak', 'sangat', 'banget', 'sekali', 'yang',
            'ini', 'itu', 'dan', 'atau', 'dengan', 'untuk', 'dari',
            'ke', 'di', 'pada', 'dalam', 'adalah', 'akan', 'sudah',
            'bisa', 'dapat', 'harus', 'saya', 'kamu', 'dia', 'kami'
        }
        self.indonesian_stopwords.update(additional_stopwords)
        
        logger.info(f"Initialized with {len(self.indonesian_stopwords)} stopwords")
    
    def get_manual_indonesian_stopwords(self):
        """Manual Indonesian stopwords list as fallback"""
        return {
            'ada', 'adalah', 'adanya', 'adapun', 'agak', 'agaknya', 'agar', 'akan',
            'akankah', 'akhir', 'akhirnya', 'aku', 'akulah', 'amat', 'amatlah',
            'anda', 'andalah', 'antar', 'antara', 'antaranya', 'apa', 'apaan',
            'apabila', 'apakah', 'apalagi', 'apatah', 'artinya', 'asal', 'asalkan',
            'atas', 'atau', 'ataukah', 'ataupun', 'awal', 'awalnya', 'bagai',
            'bagaikan', 'bagaimana', 'bagaimanakah', 'bagaimanapun', 'bagi',
            'bagian', 'bahkan', 'bahwa', 'bahwasanya', 'baik', 'bakal', 'bakalan',
            'balik', 'banyak', 'bapak', 'baru', 'bawah', 'beberapa', 'begini',
            'beginian', 'beginilah', 'begitu', 'begitulah', 'begitupun', 'bekerja',
            'belakang', 'belakangan', 'belum', 'belumlah', 'benar', 'benarkah',
            'benarlah', 'berada', 'berakhir', 'berakhirlah', 'berakhirnya',
            'berapa', 'berapakah', 'berapalah', 'berapapun', 'berarti', 'berawal',
            'berbagai', 'berdatangan', 'beri', 'berikan', 'berikut', 'berikutnya',
            'berjumlah', 'berkali-kali', 'berkata', 'berkehendak', 'berkeinginan',
            'berkenaan', 'berlainan', 'berlalu', 'berlangsung', 'berlebihan',
            'bermacam', 'bermacam-macam', 'bermaksud', 'bermula', 'bersama',
            'bersama-sama', 'bersiap', 'bersiap-siap', 'bertanya', 'bertanya-tanya',
            'berturut', 'berturut-turut', 'bertutur', 'berujar', 'berupa', 'besar',
            'betul', 'betulkah', 'biasa', 'biasanya', 'bila', 'bilakah', 'bisa',
            'bisakah', 'boleh', 'bolehkah', 'bukan', 'bukankah', 'bukanlah',
            'bukannya', 'bulan', 'bung', 'cara', 'caranya', 'cukup', 'cukupkah',
            'cukuplah', 'cuma', 'dahulu', 'dalam', 'dan', 'dapat', 'dari',
            'daripada', 'datang', 'dekat', 'demi', 'demikian', 'demikianlah',
            'dengan', 'depan', 'di', 'dia', 'diakhiri', 'diakhirinya', 'dialah',
            'diantara', 'diantaranya', 'diberi', 'diberikan', 'diberikannya',
            'dibuat', 'dibuatnya', 'didapat', 'didatangkan', 'digunakan', 'diibaratkan',
            'diibaratkannya', 'diingat', 'diingatkan', 'diinginkan', 'dijawab',
            'dijelaskan', 'dijelaskannya', 'dikarenakan', 'dikatakan', 'dikatakannya',
            'dikerjakan', 'diketahui', 'diketahuinya', 'dikira', 'dilakukan',
            'dilalui', 'dilihat', 'dimaksud', 'dimaksudkan', 'dimaksudkannya',
            'dimaksudnya', 'diminta', 'dimintai', 'dimisalkan', 'dimulai',
            'dimulainya', 'dimungkinkan', 'dini', 'dipastikan', 'diperbuat',
            'diperbuatnya', 'dipergunakan', 'diperkirakan', 'diperlihatkan',
            'diperlukan', 'diperlukannya', 'dipersoalkan', 'dipertanyakan',
            'dipunyai', 'diri', 'dirinya', 'disampaikan', 'disebut', 'disebutkan',
            'disebutkannya', 'disini', 'disinilah', 'ditambahkan', 'ditandaskan',
            'ditanya', 'ditanyai', 'ditanyakan', 'ditegaskan', 'ditujukan',
            'ditunjuk', 'ditunjuki', 'ditunjukkan', 'ditunjukkannya', 'ditunjuknya',
            'diwaktu', 'diyakini', 'dong', 'dua', 'dulu', 'empat', 'enggak',
            'enggaknya', 'entah', 'entahlah', 'guna', 'gunakan', 'hal', 'hampir',
            'hanya', 'hanyalah', 'hari', 'harus', 'haruslah', 'harusnya', 'hendak',
            'hendaklah', 'hendaknya', 'hingga', 'ia', 'ialah', 'ibarat', 'ibaratkan',
            'ibaratnya', 'ibu', 'ikut', 'ingat', 'ingat-ingat', 'ingin', 'inginkah',
            'inginkan', 'ini', 'inikah', 'inilah', 'itu', 'itukah', 'itulah',
            'jadi', 'jadilah', 'jadinya', 'jangan', 'jangankan', 'janganlah',
            'jauh', 'jawab', 'jawaban', 'jawabnya', 'jelas', 'jelaskan', 'jelaslah',
            'jelasnya', 'jika', 'jikalau', 'juga', 'jumlah', 'jumlahnya', 'justru',
            'kala', 'kalau', 'kalaulah', 'kalaupun', 'kalian', 'kami', 'kamilah',
            'kamu', 'kamulah', 'kan', 'kapan', 'kapankah', 'kapanpun', 'karena',
            'karenanya', 'kasus', 'kata', 'katakan', 'katakanlah', 'katanya',
            'ke', 'keadaan', 'kebetulan', 'kecil', 'kedua', 'keduanya', 'keinginan',
            'kelamaan', 'kelihatan', 'kelihatannya', 'keluar', 'kemana', 'kemana-mana',
            'kemari', 'kemarin', 'kemudian', 'kemungkinan', 'kemungkinannya',
            'kenapa', 'kepada', 'kepadanya', 'kesampaian', 'keseluruhan',
            'keseluruhannya', 'ketika', 'ketimbang', 'kira', 'kira-kira', 'kiranya',
            'kita', 'kitalah', 'kok', 'lagi', 'lagian', 'lah', 'lain', 'lainnya',
            'lalu', 'lama', 'lamanya', 'lebih', 'lewat', 'lima', 'luar', 'macam',
            'maka', 'makanya', 'makin', 'malah', 'malahan', 'mampu', 'mampukah',
            'mana', 'manakala', 'manalagi', 'masa', 'masalah', 'masalahnya',
            'masih', 'masihkah', 'masing', 'masing-masing', 'mau', 'maupun',
            'melainkan', 'melakukan', 'melalui', 'melihat', 'melihatnya', 'memang',
            'memastikan', 'memberi', 'memberikan', 'membuat', 'memerlukan',
            'memihak', 'meminta', 'memintakan', 'memisalkan', 'memperbuat',
            'mempergunakan', 'memperkirakan', 'memperlihatkan', 'mempersiapkan',
            'mempersoalkan', 'mempertanyakan', 'mempunyai', 'memulai', 'memungkinkan',
            'menaiki', 'menambahkan', 'menandaskan', 'menanti', 'menantikan',
            'menanya', 'menanyai', 'menanyakan', 'mendapat', 'mendapatkan',
            'mendatang', 'mendatangi', 'mendatangkan', 'menegaskan', 'mengakhiri',
            'mengapa', 'mengatakan', 'mengatakannya', 'mengenai', 'mengerjakan',
            'mengetahui', 'menggunakan', 'menghendaki', 'mengibaratkan', 'mengibaratkannya',
            'mengingat', 'mengingatkan', 'menginginkan', 'mengira', 'mengucapkan',
            'mengucapkannya', 'menjadi', 'menjawab', 'menjelaskan', 'menuju',
            'menunjuk', 'menunjuki', 'menunjukkan', 'menunjuknya', 'menurut',
            'menuturkan', 'menyampaikan', 'menyangkut', 'menyatakan', 'menyebutkan',
            'menyeluruh', 'menyiapkan', 'merasa', 'mereka', 'merekalah', 'merupakan',
            'meski', 'meskipun', 'meyakini', 'meyakinkan', 'minta', 'mirip',
            'misal', 'misalkan', 'misalnya', 'mula', 'mulai', 'mulanya', 'mungkin',
            'mungkinkah', 'nah', 'naik', 'namun', 'nanti', 'nantinya', 'nyaris',
            'oleh', 'olehnya', 'pada', 'padahal', 'padanya', 'pak', 'paling',
            'panjang', 'pantas', 'para', 'pasti', 'pastilah', 'penting', 'pentingnya',
            'per', 'percuma', 'perlu', 'perlukah', 'perlunya', 'pernah', 'persoalan',
            'pertama', 'pertama-tama', 'pertanyaan', 'pertanyakan', 'pihak',
            'pihaknya', 'pukul', 'pula', 'pun', 'punya', 'rasa', 'rasanya',
            'rata', 'rupanya', 'saat', 'saatnya', 'saja', 'sajalah', 'saling',
            'sama', 'sama-sama', 'sambil', 'sampai', 'sampai-sampai', 'sampaikan',
            'sana', 'sangat', 'sangatlah', 'satu', 'saya', 'sayalah', 'se',
            'sebab', 'sebabnya', 'sebagai', 'sebagaimana', 'sebagainya', 'sebagian',
            'sebaik', 'sebaik-baiknya', 'sebaiknya', 'sebaliknya', 'sebanyak',
            'sebegini', 'sebegitu', 'sebelum', 'sebelumnya', 'sebenarnya',
            'seberapa', 'sebesar', 'sebisanya', 'sebuah', 'sebut', 'sebutlah',
            'sebutnya', 'secara', 'secukupnya', 'sedang', 'sedangkan', 'sedemikian',
            'sedikit', 'sedikitnya', 'seenaknya', 'segala', 'segalanya', 'segera',
            'seharusnya', 'sehingga', 'seingat', 'sejak', 'sejauh', 'sejenak',
            'sejumlah', 'sekadar', 'sekadarnya', 'sekali', 'sekali-kali', 'sekalian',
            'sekaligus', 'sekalipun', 'sekarang', 'sekecil', 'seketika', 'sekiranya',
            'sekitar', 'sekitarnya', 'sekurang-kurangnya', 'sekurangnya', 'sela',
            'selain', 'selaku', 'selama', 'selama-lamanya', 'selamanya', 'selanjutnya',
            'seluruh', 'seluruhnya', 'semacam', 'semakin', 'semampu', 'semampunya',
            'semasa', 'semasih', 'semata', 'semata-mata', 'semaunya', 'sementara',
            'semisal', 'semisalnya', 'sempat', 'semua', 'semuanya', 'semula',
            'sendiri', 'sendirian', 'sendirinya', 'seolah', 'seolah-olah', 'seorang',
            'sepanjang', 'sepantasnya', 'sepantasnyalah', 'seperlunya', 'seperti',
            'sepertinya', 'sepihak', 'sering', 'seringnya', 'serta', 'serupa',
            'sesaat', 'sesama', 'sesampai', 'sesegera', 'sesekali', 'seseorang',
            'sesuatu', 'sesuatunya', 'sesudah', 'sesudahnya', 'setelah', 'setempat',
            'setengah', 'seterusnya', 'setiap', 'setiba', 'setibanya', 'setidak-tidaknya',
            'setidaknya', 'setinggi', 'seusai', 'sewajarnya', 'sewaktu', 'siap',
            'siapa', 'siapakah', 'siapapun', 'sini', 'sinilah', 'soal', 'soalnya',
            'suatu', 'sudah', 'sudahlah', 'sudahkah', 'supaya', 'tadi', 'tadinya',
            'tahu', 'tahun', 'tak', 'tambah', 'tambahnya', 'tampak', 'tampaknya',
            'tandas', 'tandasnya', 'tanpa', 'tanya', 'tanyakan', 'tanyanya',
            'tapi', 'tegas', 'tegasnya', 'telah', 'tempat', 'tengah', 'tentang',
            'tentu', 'tentulah', 'tentunya', 'tepat', 'terakhir', 'terasa',
            'terbanyak', 'terdahulu', 'terdapat', 'terdiri', 'terhadap', 'terhadapnya',
            'teringat', 'teringat-ingat', 'terjadi', 'terjadilah', 'terjadinya',
            'terkira', 'terlalu', 'terlebih', 'terlihat', 'termasuk', 'ternyata',
            'tersampaikan', 'tersebut', 'tersebutlah', 'tertentu', 'tertuju',
            'terus', 'terutama', 'tetap', 'tetapi', 'tiap', 'tiba', 'tiba-tiba',
            'tidak', 'tidakkah', 'tidaklah', 'tiga', 'tinggi', 'toh', 'tunjuk',
            'turut', 'tutur', 'tuturnya', 'ucap', 'ucapnya', 'ujar', 'ujarnya',
            'umum', 'umumnya', 'ungkap', 'ungkapnya', 'untuk', 'usah', 'usai',
            'waduh', 'wah', 'wahai', 'waktu', 'waktunya', 'walau', 'walaupun',
            'wong', 'yaitu', 'yakin', 'yakni', 'yang'
        }
    
    def create_sentiment_labels(self, df):
        """Create sentiment labels based on star ratings"""
        def rating_to_sentiment(rating):
            if pd.isna(rating):
                return 'neutral'
            rating = int(rating)
            if rating <= 2:
                return 'negative'
            elif rating == 3:
                return 'neutral'
            else:
                return 'positive'
        
        df['sentiment'] = df['score'].apply(rating_to_sentiment)
        return df
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text) or text is None or text == '':
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation and special characters, keep Indonesian characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove digits
        text = re.sub(r'\d+', '', text)
        
        return text
    
    def simple_tokenize(self, text):
        """Simple tokenization fallback when NLTK is not available"""
        if not text:
            return []
        
        # Split on whitespace and remove empty strings
        tokens = [token.strip() for token in text.split() if token.strip()]
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens 
                 if token not in self.indonesian_stopwords and len(token) > 2]
        
        return tokens
    
    def tokenize_and_filter(self, text):
        """Tokenize text and remove stopwords with fallback method"""
        if not text or text.strip() == '':
            return []
        
        if NLTK_AVAILABLE:
            try:
                # Try NLTK tokenization
                tokens = word_tokenize(text)
                
                # Remove stopwords and short tokens
                tokens = [token for token in tokens 
                         if token not in self.indonesian_stopwords and len(token) > 2]
                
                return tokens
            except Exception as e:
                logger.warning(f"NLTK tokenization failed: {e}, using simple tokenization")
                return self.simple_tokenize(text)
        else:
            # Use simple tokenization
            return self.simple_tokenize(text)
    
    def preprocess_dataframe(self, df):
        """Complete preprocessing pipeline for the dataframe"""
        logger.info("Starting data preprocessing...")
        
        # Handle missing values
        original_size = len(df)
        df = df.dropna(subset=['content', 'score'])
        logger.info(f"Dropped {original_size - len(df)} rows with missing content/score")
        
        # Create sentiment labels
        df = self.create_sentiment_labels(df)
        
        # Clean text content
        logger.info("Cleaning text content...")
        df['cleaned_content'] = df['content'].apply(self.clean_text)
        
        # Remove empty reviews after cleaning
        df = df[df['cleaned_content'].str.len() > 0]
        logger.info(f"Removed {original_size - len(df)} empty reviews after cleaning")
        
        # Tokenize
        logger.info("Tokenizing text...")
        df['tokens'] = df['cleaned_content'].apply(self.tokenize_and_filter)
        df['token_count'] = df['tokens'].apply(len)
        
        # Filter out very short reviews (less than 3 tokens)
        df = df[df['token_count'] >= 3]
        
        logger.info(f"Preprocessing complete. Final dataset size: {len(df)}")
        logger.info(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
        
        return df

if __name__ == "__main__":
    # Test the preprocessor
    import pandas as pd
    
    # Create test data
    test_data = pd.DataFrame({
        'content': [
            'Aplikasi sangat bagus sekali!',
            'Terrible app, tidak bisa digunakan',
            'Biasa saja, nothing special',
            'Suka banget sama fitur-fiturnya',
            ''
        ],
        'score': [5, 1, 3, 4, 2],
        'app_name': ['tokopedia', 'shopee', 'tokopedia', 'shopee', 'tokopedia']
    })
    
    # Test preprocessing
    preprocessor = ReviewPreprocessor()
    processed_data = preprocessor.preprocess_dataframe(test_data.copy())
    
    print("Original data:")
    print(test_data)
    print("\nProcessed data:")
    print(processed_data[['content', 'cleaned_content', 'sentiment', 'token_count']])