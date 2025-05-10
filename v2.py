import tkinter as tk
import random
import string
import json
import os
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tkinter import messagebox, ttk
from collections import defaultdict, deque
from threading import Thread
from queue import Queue

from PIL import Image, ImageTk

# Constants
MAX_LEVELS = 20

# Configuration
INITIAL_GRID_SIZE = 5
MAX_GRID_SIZE = 10
GRID_INCREMENT = 1
MAX_LEVELS = 20
WORD_DATA_DIR = "word_data"
THESAURUS_FILE = os.path.join(WORD_DATA_DIR, "thesaurus.json")
WORD_VECTORS_FILE = os.path.join(WORD_DATA_DIR, "word_vectors.json")
SYNONYM_CACHE_FILE = os.path.join(WORD_DATA_DIR, "synonym_cache.json")

class WordBankManager:
    """Handles all word-related operations including storage, retrieval, and similarity"""
    def __init__(self):
        self.word_data = defaultdict(dict)
        self.vectorizer = TfidfVectorizer()
        self.knn_model = None
        self.word_vectors = None
        self.word_list = []
        self.synonym_cache = {}
        
        self.initialize_word_bank()
    
    def initialize_word_bank(self):
        """Initialize or load word bank data"""
        self._ensure_data_directory()
        self._load_or_create_word_data()
        self._load_synonym_cache()
        self._prepare_ai_models()
    
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(WORD_DATA_DIR):
            os.makedirs(WORD_DATA_DIR)
    
    def _load_or_create_word_data(self):
        """Load existing word data or create default data"""
        if os.path.exists(THESAURUS_FILE):
            with open(THESAURUS_FILE, 'r') as f:
                self.word_data = json.load(f)
        else:
            self.word_data = {
                'animals': ['LION', 'TIGER', 'BEAR', 'WOLF', 'SHARK'],
                'fruits': ['APPLE', 'MANGO', 'GRAPE', 'PEACH', 'LEMON'],
                'countries': ['CHINA', 'INDIA', 'ITALY', 'SPAIN', 'JAPAN']
            }
            self._save_word_data()
            self.scrape_additional_words()
    
    def _load_synonym_cache(self):
        """Load synonym cache if exists"""
        if os.path.exists(SYNONYM_CACHE_FILE):
            with open(SYNONYM_CACHE_FILE, 'r') as f:
                self.synonym_cache = json.load(f)
    
    def scrape_additional_words(self):
        """Web scrape to get more words and synonyms"""
        categories = list(self.word_data.keys())
        scrape_thread = Thread(target=self._perform_scraping, args=(categories,))
        scrape_thread.daemon = True
        scrape_thread.start()
    
    def _perform_scraping(self, categories):
        """Actual web scraping implementation"""
        for category in categories:
            try:
                url = f"https://www.thesaurus.com/browse/{category}"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                synonyms = []
                container = soup.find('div', {'id': 'meanings'})
                if container:
                    items = container.find_all('a', {'class': 'css-1kg1yv8 eh475bn0'})
                    synonyms = [item.text.strip().upper() for item in items[:10]]
                
                for word in synonyms:
                    if len(word) <= MAX_GRID_SIZE and word.isalpha():
                        self.add_word(word, category)
                
                self.synonym_cache[category] = synonyms[:5]
                
            except Exception as e:
                print(f"Error scraping {category}: {e}")
        
        self._save_word_data()
        self._save_synonym_cache()
    
    def _save_word_data(self):
        """Save word data to file"""
        with open(THESAURUS_FILE, 'w') as f:
            json.dump(self.word_data, f, indent=2)
    
    def _save_synonym_cache(self):
        """Save synonym cache to file"""
        with open(SYNONYM_CACHE_FILE, 'w') as f:
            json.dump(self.synonym_cache, f, indent=2)
    
    def _prepare_ai_models(self):
        """Prepare AI models for word similarity"""
        self.word_list = []
        contexts = []
        
        for category, words in self.word_data.items():
            for word in words:
                self.word_list.append(word)
                contexts.append(f"{word} {category}")
        
        if contexts:
            self.word_vectors = self.vectorizer.fit_transform(contexts)
            self.knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
            self.knn_model.fit(self.word_vectors)
    
    def get_similar_words(self, word, category=None, n=5):
        """Get similar words using multiple methods"""
        similar_words = self._get_cached_synonyms(word, category)
        similar_words.extend(self._get_knn_similar_words(word))
        
        if len(similar_words) < n:
            similar_words.extend(self._local_search_related_words(word, n - len(similar_words)))
        
        return similar_words[:n]
    
    def _get_cached_synonyms(self, word, category):
        """Get similar words from cache"""
        if category in self.synonym_cache:
            return [w for w in self.synonym_cache[category] if w != word and w in self.word_list]
        return []
    
    def _get_knn_similar_words(self, word):
        """Get similar words using KNN"""
        if word in self.word_list:
            idx = self.word_list.index(word)
            distances, indices = self.knn_model.kneighbors(self.word_vectors[idx])
            return [self.word_list[i] for i in indices[0] if self.word_list[i] != word]
        return []
    
    def _local_search_related_words(self, word, n):
        """Find related words using local search"""
        related = []
        prefix = word[:2]
        suffix = word[-2:]
        
        for w in self.word_list:
            if w != word and (w.startswith(prefix) or w.endswith(suffix)):
                related.append(w)
                if len(related) >= n:
                    break
        return related
    
    def get_category_words(self, category, max_length=None):
        """Get words from a specific category"""
        words = self.word_data.get(category, [])
        if max_length:
            words = [w for w in words if len(w) <= max_length]
        return words
    
    def get_random_category(self):
        """Get a random category"""
        return random.choice(list(self.word_data.keys()))
    
    def add_word(self, word, category):
        """Add a new word to the word bank"""
        word = word.upper().strip()
        if not word.isalpha():
            return False
            
        if category not in self.word_data:
            self.word_data[category] = []
        
        if word not in [w.upper() for w in self.word_data[category]]:
            self.word_data[category].append(word)
            self._save_word_data()
            Thread(target=self._prepare_ai_models).start()
            return True
        return False

class WordGrid:
    """Represents the game grid with words"""
    def __init__(self, size, level, word_bank):
        self.size = min(size, MAX_GRID_SIZE)
        self.level = level
        self.word_bank = word_bank
        self.grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]
        self.words = []
        self.placed_words = []
        self.word_paths = {}
        
        self.generate_grid()
    
    def generate_grid(self):
        """Generate a new game grid with words"""
        category = self.word_bank.get_random_category()
        base_words = self.word_bank.get_category_words(category, self.size)
        
        num_words = min(self.size, max(3, 3 + (self.level // 3)))
        self.words = random.sample(base_words, min(num_words, len(base_words)))
        
        if self.level > 5:
            for word in self.words[:2]:
                similar = self.word_bank.get_similar_words(word, category, 2)
                if similar:
                    self.words.append(random.choice(similar))
        
        self.words = list(set([w for w in self.words if len(w) <= self.size]))
        self.words = random.sample(self.words, min(num_words, len(self.words)))
        
        self.place_words()
        self.fill_empty_spaces()
    
    def place_words(self):
        """Place words on the grid"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for word in self.words:
            placed = False
            attempts = 0
            
            while not placed and attempts < 100:
                direction = random.choice(directions)
                max_row = self.size - len(word) if direction[0] == 1 else self.size - 1
                max_col = self.size - len(word) if direction[1] == 1 else (
                    len(word) - 1 if direction[1] == -1 else self.size - 1
                )
                
                if max_row < 0 or max_col < 0:
                    attempts += 1
                    continue
                
                row = random.randint(0, max_row)
                col = random.randint(0, max_col)
                
                if self.can_place_word(word, row, col, direction):
                    path = []
                    for i in range(len(word)):
                        r = row + direction[0] * i
                        c = col + direction[1] * i
                        self.grid[r][c] = word[i]
                        path.append((r, c))
                    
                    self.placed_words.append(word)
                    self.word_paths[word] = path
                    placed = True
                
                attempts += 1
    
    def can_place_word(self, word, row, col, direction):
        """Check if word can be placed at given position"""
        for i in range(len(word)):
            r = row + direction[0] * i
            c = col + direction[1] * i
            
            if not (0 <= r < self.size and 0 <= c < self.size):
                return False
                
            if self.grid[r][c] != ' ' and self.grid[r][c] != word[i]:
                return False
                
        return True
    
    def find_word_path(self, word):
        """Find path of a word in the grid"""
        directions = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        
        def dfs(r, c, index, path):
            if index == len(word):
                return path
                
            if not (0 <= r < self.size and 0 <= c < self.size):
                return None
                
            if self.grid[r][c] != word[index]:
                return None
                
            for dr, dc in directions:
                new_path = path + [(r, c)]
                result = dfs(r + dr, c + dc, index + 1, new_path)
                if result:
                    return result
                    
            return None
        
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == word[0]:
                    path = dfs(i, j, 0, [])
                    if path:
                        return path
        return None
    
    def fill_empty_spaces(self):
        """Fill empty spaces with random letters"""
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == ' ':
                    self.grid[i][j] = random.choice(string.ascii_uppercase)

class LevelManager:
    """Manages game levels and progression"""
    def __init__(self, word_bank):
        self.word_bank = word_bank
        self.level = 1
        self.score = 0
        self.grid_size = INITIAL_GRID_SIZE
        self.current_grid = None
    
    def start_level(self):
        """Start a new level"""
        self.current_grid = WordGrid(self.grid_size, self.level, self.word_bank)
        return self.current_grid
    
    def level_complete(self):
        """Handle level completion"""
        bonus = self.grid_size * 20
        self.score += bonus
        
        if self.level >= MAX_LEVELS:
            return False
        
        if self.level % 3 == 0 and self.grid_size < MAX_GRID_SIZE:
            self.grid_size += GRID_INCREMENT
            
        self.level += 1
        return True
    
    def reset(self):
        """Reset game to initial state"""
        self.level = 1
        self.score = 0
        self.grid_size = INITIAL_GRID_SIZE
        self.current_grid = None

class AIPlayer:
    """AI player that can compete against humans"""
    def __init__(self, word_bank):
        self.word_bank = word_bank
        self.difficulty = 1  # 1-5 scale
    
    def make_move(self, grid):
        """Make a move based on current grid state"""
        if not grid.placed_words:
            return None
        
        # Different strategies based on difficulty
        if self.difficulty < 3:
            # Easy: random word
            return random.choice(grid.placed_words)
        elif self.difficulty < 5:
            # Medium: longest word
            return max(grid.placed_words, key=len)
        else:
            # Hard: uses word similarity for better choices
            category = self.word_bank.get_random_category()
            for word in sorted(grid.placed_words, key=len, reverse=True):
                similar = self.word_bank.get_similar_words(word, category, 1)
                if similar and similar[0] in grid.placed_words:
                    return similar[0]
            return max(grid.placed_words, key=len)
    
    def adjust_difficulty(self, level):
        """Adjust AI difficulty based on game level"""
        self.difficulty = min(5, max(1, level // 4))

class TwoPlayerGame:
    """Manages a game between two human players"""
    def __init__(self, word_bank):
        self.word_bank = word_bank
        self.reset()
    
    def reset(self):
        """Reset the game state"""
        self.level = 1
        self.grid_size = INITIAL_GRID_SIZE
        self.current_grid = None
        self.current_player = 1  # Player 1 or 2
        self.scores = {1: 0, 2: 0}
    
    def start_level(self):
        """Start a new level"""
        self.current_grid = WordGrid(self.grid_size, self.level, self.word_bank)
        return self.current_grid
    
    def make_move(self, player, word):
        """Handle a player's move"""
        if word in self.current_grid.placed_words:
            self.current_grid.placed_words.remove(word)
            self.scores[player] += len(word) * 10
            self.current_player = 2 if player == 1 else 1
            return True
        return False
    
    def level_complete(self):
        """Check if level is complete and handle progression"""
        if not self.current_grid.placed_words:
            if self.level % 3 == 0 and self.grid_size < MAX_GRID_SIZE:
                self.grid_size += GRID_INCREMENT
            self.level += 1
            return True
        return False


import tkinter as tk
from tkinter import messagebox, ttk
from collections import deque
import random
import os
from PIL import Image, ImageTk

# Constants
MAX_LEVELS = 20

# Theme colors dictionary
THEME_COLORS = {
    'animals': {
        'bg': '#f0f8ff',  # Light blue background
        'title': '#1e3d59',  # Dark blue title
        'button': '#43b0f1',  # Medium blue buttons
        'grid': '#e8f4fc',  # Very light blue grid
        'highlight': '#a2d5f2'  # Highlighted cells
    },
    'fruits': {
        'bg': '#fff8f0',  # Light peach background
        'title': '#ff6f3c',  # Orange title
        'button': '#ff9a3c',  # Medium orange buttons
        'grid': '#fce8e8',  # Very light peach grid
        'highlight': '#ffba93'  # Highlighted cells
    },
    'countries': {
        'bg': '#f0fff8',  # Light mint background
        'title': '#1b512d',  # Dark green title
        'button': '#4caf50',  # Medium green buttons
        'grid': '#e8fcf4',  # Very light mint grid
        'highlight': '#a2f2d5'  # Highlighted cells
    },
    'default': {
        'bg': '#f0f4f8',  # Original background
        'title': '#374785',  # Original title
        'button': '#70c1b3',  # Original buttons
        'grid': 'white',    # Original grid
        'highlight': '#ace7ef'  # Original highlighted cells
    }
}

class GameUI:
    """Handles all user interface components"""
    def __init__(self, root, game_controller):
        self.root = root
        self.controller = game_controller
        self.selected_cells = []
        self.current_theme = "default"
        
        self.setup_ui()
    
    def setup_ui(self):
        """Initialize all UI components"""
        self.root.configure(bg=THEME_COLORS[self.current_theme]['bg'])
        
        self.main_frame = tk.Frame(self.root, padx=20, pady=20, bg=THEME_COLORS[self.current_theme]['bg'])
        self.main_frame.pack(expand=True, fill=tk.BOTH)
        
        self._setup_header()
        self._setup_theme_selector()
        self._setup_options_panel()
        self._setup_info_panel()
        self._setup_grid()
        self._setup_input_area()
        #self._setup_word_list()
    
    def _setup_header(self):
        """Setup header section"""
        self.header_frame = tk.Frame(self.main_frame, bg=THEME_COLORS[self.current_theme]['bg'])
        self.header_frame.pack(fill=tk.X)
        
        self.title_label = tk.Label(
            self.header_frame, 
            text="🚀 WORD HUNT CHALLENGE 🚀", 
            font=('Comic Sans MS', 28, 'bold'),
            fg=THEME_COLORS[self.current_theme]['title'],
            bg=THEME_COLORS[self.current_theme]['bg']
        )
        self.title_label.pack(pady=10)
    
    def _setup_theme_selector(self):
        """Setup theme selection dropdown"""
        self.theme_frame = tk.Frame(self.main_frame, bg=THEME_COLORS[self.current_theme]['bg'])
        self.theme_frame.pack(fill=tk.X, pady=5)
        
        self.theme_label = tk.Label(
            self.theme_frame,
            text="Select Theme:",
            font=('Comic Sans MS', 14, 'bold'),
            bg=THEME_COLORS[self.current_theme]['bg'],
            fg=THEME_COLORS[self.current_theme]['title']
        )
        self.theme_label.pack(side=tk.LEFT, padx=5)
        
        self.theme_var = tk.StringVar()
        self.theme_var.set("default")
        
        self.theme_menu = ttk.Combobox(
            self.theme_frame,
            textvariable=self.theme_var,
            values=["default", "animals", "fruits", "countries"],
            font=('Comic Sans MS', 12),
            width=12,
            state="readonly"
        )
        self.theme_menu.pack(side=tk.LEFT, padx=5)
        self.theme_menu.bind("<<ComboboxSelected>>", self.change_theme)
        
        self.play_theme_button = tk.Button(
            self.theme_frame,
            text="Play With Theme",
            command=self.load_themed_game,
            font=('Comic Sans MS', 12, 'bold'),
            bg=THEME_COLORS[self.current_theme]['button'],
            fg='white',
            activebackground='#247ba0',
            activeforeground='white',
            bd=0,
            relief='ridge',
            cursor='hand2'
        )
        self.play_theme_button.pack(side=tk.LEFT, padx=15)
    
    def _setup_options_panel(self):
        """Setup panel with 5 main options"""
        self.options_frame = tk.Frame(self.main_frame, bg=THEME_COLORS[self.current_theme]['bg'])
        self.options_frame.pack(pady=10)
        
        button_style = {
            'font': ('Comic Sans MS', 12, 'bold'),
            'bg': THEME_COLORS[self.current_theme]['button'],
            'fg': 'white',
            'activebackground': '#247ba0',
            'activeforeground': 'white',
            'bd': 0,
            'relief': 'ridge',
            'width': 14,
            'height': 2,
            'cursor': 'hand2'
        }
        
        # Hint Button
        self.hint_button = tk.Button(
            self.options_frame,
            text="🎯 Hint",
            command=self.controller.give_hint,
            **button_style
        )
        self.hint_button.grid(row=0, column=0, padx=10, pady=5)
        
        # New Game Button
        self.new_game_button = tk.Button(
            self.options_frame,
            text="🆕 New Game",
            command=self.controller.new_game,
            **button_style
        )
        self.new_game_button.grid(row=0, column=1, padx=10, pady=5)
        
        # Single Player
        self.single_player_button = tk.Button(
            self.options_frame,
            text="👤 Single Player",
            command=lambda: self.controller.set_game_mode('single'),
            **button_style
        )
        self.single_player_button.grid(row=1, column=0, padx=10, pady=5)
        
        # Two Player
        self.two_player_button = tk.Button(
            self.options_frame,
            text="👥 Two Player",
            command=lambda: self.controller.set_game_mode('two_player'),
            **button_style
        )
        self.two_player_button.grid(row=1, column=1, padx=10, pady=5)
        
        # VS AI
        self.vs_ai_button = tk.Button(
            self.options_frame,
            text="🤖 VS AI",
            command=lambda: self.controller.set_game_mode('ai'),
            **button_style
        )
        self.vs_ai_button.grid(row=2, column=0, columnspan=2, pady=5)
    
    def _setup_info_panel(self):
        """Setup level and score display"""
        self.info_frame = tk.Frame(self.main_frame, bg=THEME_COLORS[self.current_theme]['bg'])
        self.info_frame.pack(fill=tk.X, pady=10)
        
        self.level_label = tk.Label(
            self.info_frame, 
            text=f"Level: {self.controller.get_level()}/{MAX_LEVELS}",
            font=('Helvetica', 14, 'bold'),
            bg=THEME_COLORS[self.current_theme]['bg'],
            fg='#333'
        )
        self.level_label.pack(side=tk.LEFT, padx=20)
        
        self.score_label = tk.Label(
            self.info_frame, 
            text=f"Score: {self.controller.get_score()}",
            font=('Helvetica', 14, 'bold'),
            bg=THEME_COLORS[self.current_theme]['bg'],
            fg='#333'
        )
        self.score_label.pack(side=tk.LEFT, padx=20)
        
        self.theme_info_label = tk.Label(
            self.info_frame,
            text="",
            font=('Helvetica', 14, 'bold'),
            bg=THEME_COLORS[self.current_theme]['bg'],
            fg='#333'
        )
        self.theme_info_label.pack(side=tk.LEFT, padx=20)
        
        self.player_turn_label = tk.Label(
            self.info_frame,
            text="",
            font=('Helvetica', 14, 'bold'),
            bg=THEME_COLORS[self.current_theme]['bg']
        )
        self.player_turn_label.pack(side=tk.RIGHT, padx=20)
    
    def _setup_grid(self):
        """Setup game grid"""
        self.grid_frame = tk.Frame(self.main_frame, bg=THEME_COLORS[self.current_theme]['bg'])
        self.grid_frame.pack(pady=20)
        self.cells = []
    
    def _setup_input_area(self):
        """Setup word input area"""
        self.input_frame = tk.Frame(self.main_frame, bg=THEME_COLORS[self.current_theme]['bg'])
        self.input_frame.pack(pady=10)
        
        self.word_entry = tk.Entry(
            self.input_frame,
            font=('Comic Sans MS', 18),
            width=20,
            relief='solid',
            bd=2
        )
        self.word_entry.pack(side=tk.LEFT, padx=5)
        self.word_entry.bind('<Return>', lambda e: self.controller.check_word())
        
        self.submit_btn = tk.Button(
            self.input_frame,
            text="✅ Submit",
            command=self.controller.check_word,
            font=('Comic Sans MS', 12, 'bold'),
            bg='#ff6f61',
            fg='white',
            activebackground='#ff3b2e',
            activeforeground='white',
            bd=0,
            relief='ridge',
            width=12,
            height=2,
            cursor='hand2'
        )
        self.submit_btn.pack(side=tk.LEFT, padx=5)
    
    def _setup_word_list(self):
        """Setup list of words to find"""
        self.words_frame = tk.Frame(self.main_frame, bg=THEME_COLORS[self.current_theme]['bg'])
        self.words_frame.pack(fill=tk.BOTH, expand=True)
        
        self.words_label = tk.Label(
            self.words_frame,
            text="📝 Words to find:",
            font=('Comic Sans MS', 14, 'bold'),
            bg=THEME_COLORS[self.current_theme]['bg'],
            fg=THEME_COLORS[self.current_theme]['title']
        )
        self.words_label.pack()
        
        self.words_list = tk.Listbox(
            self.words_frame,
            font=('Comic Sans MS', 14),
            bg='#ffffff',
            bd=2,
            relief='solid',
            height=6
        )
        self.words_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_grid(self, grid_data):
        """Create the game grid UI"""
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        
        self.cells = []
        for i in range(len(grid_data)):
            row = []
            for j in range(len(grid_data[i])):
                cell = tk.Label(
                    self.grid_frame,
                    text=grid_data[i][j],
                    font=('Comic Sans MS', 20, 'bold'),
                    width=3,
                    height=1,
                    relief='groove',
                    bg=THEME_COLORS[self.current_theme]['grid']
                )
                cell.grid(row=i, column=j, padx=3, pady=3)
                cell.bind('<Button-1>', lambda e, i=i, j=j: self.select_cell(i, j))
                row.append(cell)
            self.cells.append(row)
    
    def select_cell(self, row, col):
        """Handle cell selection"""
        self.selected_cells.append((row, col))
        self.word_entry.insert(tk.END, self.controller.get_grid_cell(row, col))
        self.cells[row][col].config(bg=THEME_COLORS[self.current_theme]['highlight'])
        
        if len(self.selected_cells) > 1:
            self.validate_path()
    
    def validate_path(self):
        """Validate selected path"""
        if len(self.selected_cells) < 2:
            return True
            
        visited = set()
        queue = deque([self.selected_cells[0]])
        target = self.selected_cells[-1]
        
        while queue:
            current = queue.popleft()
            if current == target:
                return True
                
            if current in visited:
                continue
                
            visited.add(current)
            
            for dr, dc in [(1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                if neighbor in self.selected_cells and neighbor not in visited:
                    queue.append(neighbor)
        
        self.clear_selection()
        return False
    
    def clear_selection(self):
        """Clear selected cells"""
        for row, col in self.selected_cells:
            self.cells[row][col].config(bg=THEME_COLORS[self.current_theme]['grid'])
        self.selected_cells = []
        self.word_entry.delete(0, tk.END)
    
    def highlight_word(self, word, path, player=None):
        """Highlight a found word on the grid"""
        color = '#90ee90'  # Default green
        if player == 1:
            color = '#b3e5fc'  # Player 1 color
        elif player == 2:
            color = '#f8bbd0'  # Player 2 color
            
        for row, col in path:
            self.cells[row][col].config(bg=color)
    
    def update_display(self, level, score, words_to_find, current_player=None):
        """Update the game display"""
        self.level_label.config(text=f"Level: {level}/{MAX_LEVELS}")
        
        if isinstance(score, dict):
            self.score_label.config(text=f"P1: {score.get(1, 0)} | P2: {score.get(2, 0)}")
        else:
            self.score_label.config(text=f"Score: {score}")
        
        if self.current_theme != "default":
            self.theme_info_label.config(text=f"Theme: {self.current_theme.capitalize()}")
        else:
            self.theme_info_label.config(text="")
        
        if current_player:
            self.player_turn_label.config(text=f"Player {current_player}'s Turn")
            self.player_turn_label.config(fg='#003366')
        else:
            self.player_turn_label.config(text="")
    
    def show_message(self, title, message):
        """Show a message box"""
        messagebox.showinfo(title, message)
    def change_theme(self, event=None):
        """Change UI theme based on selected theme"""
        new_theme = self.theme_var.get()
        
        # Update the current theme
        old_theme = self.current_theme
        self.current_theme = new_theme
        self.controller.current_theme = new_theme  # Make sure controller knows about the theme change
        
        # Update background colors
        self.root.configure(bg=THEME_COLORS[self.current_theme]['bg'])
        self.main_frame.configure(bg=THEME_COLORS[self.current_theme]['bg'])
        self.header_frame.configure(bg=THEME_COLORS[self.current_theme]['bg'])
        self.theme_frame.configure(bg=THEME_COLORS[self.current_theme]['bg'])
        self.options_frame.configure(bg=THEME_COLORS[self.current_theme]['bg'])
        self.info_frame.configure(bg=THEME_COLORS[self.current_theme]['bg'])
        self.grid_frame.configure(bg=THEME_COLORS[self.current_theme]['bg'])
        self.input_frame.configure(bg=THEME_COLORS[self.current_theme]['bg'])
        
        # Update labels
        self.title_label.configure(
            fg=THEME_COLORS[self.current_theme]['title'],
            bg=THEME_COLORS[self.current_theme]['bg']
        )
        self.theme_label.configure(
            bg=THEME_COLORS[self.current_theme]['bg'],
            fg=THEME_COLORS[self.current_theme]['title']
        )
        self.level_label.configure(bg=THEME_COLORS[self.current_theme]['bg'])
        self.score_label.configure(bg=THEME_COLORS[self.current_theme]['bg'])
        self.theme_info_label.configure(bg=THEME_COLORS[self.current_theme]['bg'])
        self.player_turn_label.configure(bg=THEME_COLORS[self.current_theme]['bg'])
        
        # Update buttons
        button_widgets = [
            self.hint_button, self.new_game_button, 
            self.single_player_button, self.two_player_button, 
            self.vs_ai_button, self.play_theme_button
        ]
        
        for button in button_widgets:
            button.configure(bg=THEME_COLORS[self.current_theme]['button'])
        
        # Update grid if it exists
        if self.cells:
            for row in self.cells:
                for cell in row:
                    if cell.cget('bg') != '#90ee90' and cell.cget('bg') != '#b3e5fc' and cell.cget('bg') != '#f8bbd0':
                        cell.configure(bg=THEME_COLORS[self.current_theme]['grid'])
        
        # Update theme info label
        if self.current_theme != "default":
            self.theme_info_label.config(text=f"Theme: {self.current_theme.capitalize()}")
        else:
            self.theme_info_label.config(text="")
        
        # Important: Force recreation of the game board with new theme words
        if old_theme != new_theme:  # Only regenerate if theme actually changed
            # For any theme change, just trigger a new game - the controller should respect the current_theme
            self.controller.new_game()
        
        # Force update the display
        self.root.update_idletasks()
        
    def load_themed_game(self):
        """Load a new game with the selected theme"""
        theme = self.theme_var.get()
        if theme == "default":
            self.show_message("Theme Selection", "Please select a specific theme (animals, fruits, countries).")
            return
            
        # Call the controller method to load themed words and start a game
        self.controller.start_themed_game(theme)
class WordHuntGame:
    """Main game controller class"""
    def __init__(self, root):
        self.root = root
        self.root.title("Word Hunt")

        self.game_mode = 'single'  # 'single', 'two_player', or 'ai'
        self.ai_turn = False
        
        self.word_bank = WordBankManager()
        self.level_manager = LevelManager(self.word_bank)
        self.two_player_game = TwoPlayerGame(self.word_bank)
        self.ai_player = AIPlayer(self.word_bank)
        self.ui = GameUI(root, self)
        
        
        
        self.new_game()

        self.root.after(100, self.new_game)
    
    def set_game_mode(self, mode):
        """Set the game mode (single player, two player, vs AI)"""
        self.game_mode = mode
        self.new_game()
    
    def new_game(self):
        """Start a new game"""
        if self.game_mode == 'single':
            self.level_manager.reset()
            self.start_level()
        elif self.game_mode == 'two_player':
            self.two_player_game.reset()
            self.start_two_player_level()
        elif self.game_mode == 'ai':
            self.level_manager.reset()
            self.ai_player.adjust_difficulty(1)
            self.start_ai_game()
    
    def start_level(self):
        """Start a new level in single player mode"""
        # Generate the game grid
        grid = self.level_manager.start_level()
        
        # Create the visual grid in the UI
        if hasattr(self.ui, 'grid_frame'):  # Safety check
            self.ui.create_grid(grid.grid)
        
        # Update the display with current game state
        self.ui.update_display(
            self.level_manager.level,
            self.level_manager.score,
            grid.placed_words
        )
    
    def start_two_player_level(self):
        """Start a new level in two player mode"""
        grid = self.two_player_game.start_level()
        self.ui.create_grid(grid.grid)
        self.ui.update_display(
            self.two_player_game.level,
            self.two_player_game.scores,
            grid.placed_words,
            self.two_player_game.current_player
        )
    
    def start_ai_game(self):
        """Start a new game against AI"""
        self.ai_turn = False
        self.start_level()
    
    def check_word(self):
        """Check if guessed word is correct"""
        guess = self.ui.word_entry.get().strip().upper()
        self.ui.clear_selection()
        
        if not guess:
            return
            
        if self.game_mode == 'single':
            self._check_single_player_word(guess)
        elif self.game_mode == 'two_player':
            self._check_two_player_word(guess)
        elif self.game_mode == 'ai':
            self._check_ai_game_word(guess)
    
    def _check_single_player_word(self, guess):
        """Check word in single player mode"""
        grid = self.level_manager.current_grid
        if guess in grid.placed_words:
            grid.placed_words.remove(guess)
            self.level_manager.score += len(guess) * 10
            self.ui.highlight_word(guess, grid.word_paths[guess])
            
            if not grid.placed_words:
                if self.level_manager.level_complete():
                    self.ui.show_message(
                        "Level Complete!", 
                        f"Bonus: {self.level_manager.grid_size * 20} points!"
                    )
                    self.start_level()
                else:
                    self.ui.show_message(
                        "Game Complete!", 
                        f"Final Score: {self.level_manager.score}"
                    )
                    self.new_game()
        
        self.ui.update_display(
            self.level_manager.level,
            self.level_manager.score,
            grid.placed_words
        )
    
    def _check_two_player_word(self, guess):
        """Check word in two player mode"""
        current_player = self.two_player_game.current_player
        grid = self.two_player_game.current_grid
        
        if self.two_player_game.make_move(current_player, guess):
            self.ui.highlight_word(guess, grid.word_paths[guess], current_player)
            
            if not grid.placed_words:
                if self.two_player_game.level_complete():
                    self.ui.show_message(
                        "Level Complete!",
                        f"Player 1: {self.two_player_game.scores[1]} | Player 2: {self.two_player_game.scores[2]}"
                    )
                    self.start_two_player_level()
                else:
                    winner = 1 if self.two_player_game.scores[1] > self.two_player_game.scores[2] else 2
                    self.ui.show_message(
                        "Game Complete!",
                        f"Player {winner} wins!\nFinal Scores:\nPlayer 1: {self.two_player_game.scores[1]}\nPlayer 2: {self.two_player_game.scores[2]}"
                    )
                    self.new_game()
        
        self.ui.update_display(
            self.two_player_game.level,
            self.two_player_game.scores,
            grid.placed_words,
            self.two_player_game.current_player
        )
    
    def _check_ai_game_word(self, guess):
        """Check word in AI mode"""
        grid = self.level_manager.current_grid
        if guess in grid.placed_words:
            grid.placed_words.remove(guess)
            self.level_manager.score += len(guess) * 10
            self.ui.highlight_word(guess, grid.word_paths[guess])
            
            if not grid.placed_words:
                if self.level_manager.level_complete():
                    self.ui.show_message(
                        "Level Complete!", 
                        f"Bonus: {self.level_manager.grid_size * 20} points!"
                    )
                    self.start_level()
                else:
                    self.ui.show_message(
                        "Game Complete!", 
                        f"Final Score: {self.level_manager.score}"
                    )
                    self.new_game()
            else:
                # AI's turn
                self.ai_turn = True
                self.root.after(1500, self.ai_make_move)
        
        self.ui.update_display(
            self.level_manager.level,
            self.level_manager.score,
            grid.placed_words
        )
    
    def ai_make_move(self):
        """Handle AI's move"""
        if not self.ai_turn or self.game_mode != 'ai':
            return
            
        grid = self.level_manager.current_grid
        if not grid.placed_words:
            return
            
        self.ai_player.adjust_difficulty(self.level_manager.level)
        ai_guess = self.ai_player.make_move(grid)
        
        if ai_guess and ai_guess in grid.placed_words:
            grid.placed_words.remove(ai_guess)
            self.ui.highlight_word(ai_guess, grid.word_paths[ai_guess])
            self.ui.show_message("AI Found", f"The AI found: {ai_guess}")
            
            if not grid.placed_words:
                if self.level_manager.level_complete():
                    self.ui.show_message(
                        "Level Complete!", 
                        f"Bonus: {self.level_manager.grid_size * 20} points!"
                    )
                    self.start_level()
                else:
                    self.ui.show_message(
                        "Game Complete!", 
                        f"Final Score: {self.level_manager.score}"
                    )
                    self.new_game()
        
        self.ai_turn = False
        self.ui.update_display(
            self.level_manager.level,
            self.level_manager.score,
            grid.placed_words
        )
    
    def give_hint(self):
        """Provide a hint to the player"""
        if self.game_mode == 'single':
            grid = self.level_manager.current_grid
        elif self.game_mode == 'two_player':
            grid = self.two_player_game.current_grid
        elif self.game_mode == 'ai':
            grid = self.level_manager.current_grid
            if self.ai_turn:
                return
        
        if not grid.placed_words:
            self.ui.show_message("Hint", "No words left to find!")
            return
            
        word = random.choice(grid.placed_words)
        hint = f"Look for a {len(word)}-letter word: {word[0]}...{word[-1]}"
        
        if self.level_manager.level > 5:
            similar = self.word_bank.get_similar_words(word)
            if similar:
                hint += f"\nRelated: {', '.join(similar[:2])}"
        
        self.ui.show_message("Hint", hint)
    
    def get_level(self):
        """Get current level"""
        if self.game_mode == 'single' or self.game_mode == 'ai':
            return self.level_manager.level
        elif self.game_mode == 'two_player':
            return self.two_player_game.level
    
    def get_score(self):
        """Get current score"""
        if self.game_mode == 'single' or self.game_mode == 'ai':
            return self.level_manager.score
        elif self.game_mode == 'two_player':
            return self.two_player_game.scores
    
    def get_grid_cell(self, row, col):
        """Get value of a grid cell"""
        if self.game_mode == 'single' or self.game_mode == 'ai':
            return self.level_manager.current_grid.grid[row][col]
        elif self.game_mode == 'two_player':
            return self.two_player_game.current_grid.grid[row][col]

if __name__ == "__main__":
    root = tk.Tk()
    game = WordHuntGame(root)
    root.mainloop()