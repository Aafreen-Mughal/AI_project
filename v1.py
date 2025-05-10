import tkinter as tk
import random
import string
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tkinter import messagebox
from collections import defaultdict

# Configuration
INITIAL_GRID_SIZE = 5
MAX_GRID_SIZE = 10
GRID_INCREMENT = 1
MAX_LEVELS = 20
WORD_DATA_DIR = "word_data"
THESAURUS_FILE = os.path.join(WORD_DATA_DIR, "thesaurus.json")
WORD_VECTORS_FILE = os.path.join(WORD_DATA_DIR, "word_vectors.json")

class WordBank:
    def __init__(self):
        self.word_data = defaultdict(dict)
        self.vectorizer = TfidfVectorizer()
        self.knn_model = None
        self.rocchio_model = None
        self.word_vectors = None
        self.word_list = []
        
        self.initialize_word_bank()
    
    def initialize_word_bank(self):
        # Create directory if it doesn't exist
        if not os.path.exists(WORD_DATA_DIR):
            os.makedirs(WORD_DATA_DIR)
        
        # Load or create thesaurus data
        if os.path.exists(THESAURUS_FILE):
            with open(THESAURUS_FILE, 'r') as f:
                self.word_data = json.load(f)
        else:
            # Default word bank if no thesaurus file exists
            default_words = {
                'animals': ['LION', 'TIGER', 'BEAR', 'WOLF', 'SHARK', 'ZEBRA', 'PANDA', 'EAGLE', 'SNAKE', 'MOUSE'],
                'fruits': ['APPLE', 'MANGO', 'GRAPE', 'PEACH', 'LEMON', 'MELON', 'BERRY', 'PRUNE', 'PLUM', 'KIWI'],
                'countries': ['CHINA', 'INDIA', 'ITALY', 'SPAIN', 'JAPAN', 'BRAZIL', 'EGYPT', 'CANADA', 'GREECE', 'CHILE'],
                'sports': ['SOCCER', 'TENNIS', 'HOCKEY', 'GOLF', 'RUGBY', 'CRICKET', 'BOXING', 'SWIMMING', 'CYCLING', 'SKIING'],
                'science': ['ATOM', 'CELL', 'GENE', 'MATH', 'DATA', 'ACID', 'BASE', 'MOON', 'STAR', 'WAVE']
            }
            self.word_data = default_words
            self.save_word_data()
        
        # Prepare word vectors and models
        self.prepare_ai_models()
    
    def save_word_data(self):
        with open(THESAURUS_FILE, 'w') as f:
            json.dump(self.word_data, f, indent=2)
    
    def prepare_ai_models(self):
        # Create a list of all words with their categories as context
        self.word_list = []
        contexts = []
        
        for category, words in self.word_data.items():
            for word in words:
                self.word_list.append(word)
                contexts.append(f"{word} {category}")
        
        # Create TF-IDF vectors
        self.word_vectors = self.vectorizer.fit_transform(contexts)
        
        # Train KNN model
        self.knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.knn_model.fit(self.word_vectors)
        
        # For Rocchio, we'll use the vectorizer directly during query time
    
    def get_similar_words(self, word, category=None, n=5):
        """Get similar words using KNN and Rocchio algorithms"""
        if word not in self.word_list:
            return []
        
        # Find index of the word
        idx = self.word_list.index(word)
        
        # KNN approach
        distances, indices = self.knn_model.kneighbors(self.word_vectors[idx])
        knn_words = [self.word_list[i] for i in indices[0] if self.word_list[i] != word]
        
        # Rocchio approach (pseudo-relevance feedback)
        query_vec = self.word_vectors[idx]
        
        # Find most similar words to the query (positive examples)
        sim_scores = (query_vec * self.word_vectors.T).toarray()[0]
        top_indices = np.argsort(sim_scores)[-n-1:-1][::-1]
        rocchio_words = [self.word_list[i] for i in top_indices if self.word_list[i] != word]
        
        # Combine results, prioritizing Rocchio then KNN
        combined = []
        seen = set()
        
        for w in rocchio_words + knn_words:
            if w not in seen and w != word:
                seen.add(w)
                combined.append(w)
                if len(combined) >= n:
                    break
        
        return combined
    
    def get_category_words(self, category, max_length=None):
        """Get words from a specific category, optionally filtered by length"""
        words = self.word_data.get(category, [])
        if max_length:
            words = [w for w in words if len(w) <= max_length]
        return words
    
    def get_random_category(self):
        """Get a random category from the available ones"""
        return random.choice(list(self.word_data.keys()))
    
    def add_word(self, word, category):
        """Add a new word to the word bank"""
        if category not in self.word_data:
            self.word_data[category] = []
        
        if word.upper() not in [w.upper() for w in self.word_data[category]]:
            self.word_data[category].append(word.upper())
            self.save_word_data()
            self.prepare_ai_models()  # Rebuild models with new data
            return True
        return False

class WordGrid:
    def __init__(self, size, level, word_bank):
        self.size = min(size, MAX_GRID_SIZE)
        self.level = level
        self.word_bank = word_bank
        self.grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]
        self.words = []
        self.placed_words = []
        
        self.generate_grid()
    
    def generate_grid(self):
        # Select word category based on level
        categories = list(self.word_bank.word_data.keys())
        selected_category = categories[(self.level - 1) % len(categories)]
        
        # Get words that fit in current grid, with some similar words mixed in
        base_words = self.word_bank.get_category_words(selected_category, self.size)
        
        # Number of words increases with level
        num_words = min(self.size, max(3, 3 + (self.level // 3)))
        
        # Select base words
        self.words = random.sample(base_words, min(num_words, len(base_words)))
        
        # For higher levels, add some similar words to make it more challenging
        if self.level > 5:
            similar_words = []
            for word in self.words[:2]:  # Get similar words for first 2 words
                similar_words.extend(self.word_bank.get_similar_words(word, selected_category, 2))
            
            # Add 1-2 similar words to the mix
            if similar_words:
                self.words.extend(random.sample(similar_words, min(2, len(similar_words))))
        
        # Remove duplicates and ensure words fit
        self.words = list(set([w for w in self.words if len(w) <= self.size]))
        self.words = random.sample(self.words, min(num_words, len(self.words)))
        
        self.place_words()
        self.fill_empty_spaces()
    
    def place_words(self):
        directions = [
            (1, 0),   # Down
            (0, 1),    # Right
            (1, 1),    # Down-right
            (1, -1)    # Down-left
        ]
        
        for word in self.words:
            word_added = False
            attempts = 0
            
            while not word_added and attempts < 100:
                direction = random.choice(directions)
                word_len = len(word)
                
                # Calculate maximum starting position
                if direction[0] == 1:  # Vertical component
                    max_row = self.size - word_len
                else:
                    max_row = self.size - 1
                
                if direction[1] == 1:  # Horizontal right
                    max_col = self.size - word_len
                elif direction[1] == -1:  # Horizontal left
                    max_col = word_len - 1
                else:
                    max_col = self.size - 1
                
                if max_row < 0 or max_col < 0:
                    attempts += 1
                    continue
                
                row = random.randint(0, max_row)
                col = random.randint(0, max_col)
                
                # Check if word can fit
                can_place = True
                for i in range(word_len):
                    r = row + direction[0] * i
                    c = col + direction[1] * i
                    if self.grid[r][c] != ' ' and self.grid[r][c] != word[i]:
                        can_place = False
                        break
                
                # Place the word
                if can_place:
                    for i in range(word_len):
                        r = row + direction[0] * i
                        c = col + direction[1] * i
                        self.grid[r][c] = word[i]
                    self.placed_words.append(word)
                    word_added = True
                
                attempts += 1
    
    def fill_empty_spaces(self):
        # First pass: try to create small valid words (2-3 letters)
        small_words = ['AT', 'IT', 'IN', 'ON', 'AN', 'OR', 'AND', 'THE', 'FOR', 'ARE']
        
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == ' ':
                    # Try horizontal
                    if j < self.size - 1 and self.grid[i][j+1] == ' ':
                        for word in small_words:
                            if len(word) == 2:
                                self.grid[i][j] = word[0]
                                self.grid[i][j+1] = word[1]
                                break
                    # Try vertical
                    elif i < self.size - 1 and self.grid[i+1][j] == ' ':
                        for word in small_words:
                            if len(word) == 2:
                                self.grid[i][j] = word[0]
                                self.grid[i+1][j] = word[1]
                                break
        
        # Second pass: fill remaining spaces with letters that form common prefixes/suffixes
        common_letters = {
            'start': ['B', 'C', 'D', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T'],
            'end': ['D', 'E', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'Y']
        }
        
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == ' ':
                    # Check surrounding letters to make better choices
                    neighbors = []
                    if i > 0: neighbors.append(self.grid[i-1][j])
                    if i < self.size-1: neighbors.append(self.grid[i+1][j])
                    if j > 0: neighbors.append(self.grid[i][j-1])
                    if j < self.size-1: neighbors.append(self.grid[i][j+1])
                    
                    # Try to form common letter combinations
                    if neighbors:
                        for letter in neighbors:
                            if letter in common_letters['start']:
                                self.grid[i][j] = random.choice(common_letters['end'])
                                break
                            elif letter in common_letters['end']:
                                self.grid[i][j] = random.choice(common_letters['start'])
                                break
                        else:
                            self.grid[i][j] = random.choice(string.ascii_uppercase)
                    else:
                        self.grid[i][j] = random.choice(string.ascii_uppercase)

class WordHuntGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Word Hunt Challenge")
        self.level = 1
        self.score = 0
        self.grid_size = INITIAL_GRID_SIZE
        self.current_grid = None
        self.word_bank = WordBank()
        
        self.setup_ui()
        self.new_game()
    
    def setup_ui(self):
        # Main container
        self.main_frame = tk.Frame(self.root, padx=20, pady=20)
        self.main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Header
        self.header_frame = tk.Frame(self.main_frame)
        self.header_frame.pack(fill=tk.X)
        
        self.title_label = tk.Label(
            self.header_frame, 
            text="WORD HUNT", 
            font=('Arial', 24, 'bold'),
            fg='navy'
        )
        self.title_label.pack(pady=10)
        
        # Info panel
        self.info_frame = tk.Frame(self.main_frame)
        self.info_frame.pack(fill=tk.X, pady=10)
        
        self.level_label = tk.Label(
            self.info_frame, 
            text=f"Level: {self.level}/{MAX_LEVELS}",
            font=('Arial', 12)
        )
        self.level_label.pack(side=tk.LEFT, padx=20)
        
        self.score_label = tk.Label(
            self.info_frame, 
            text=f"Score: {self.score}",
            font=('Arial', 12)
        )
        self.score_label.pack(side=tk.LEFT, padx=20)
        
        self.grid_label = tk.Label(
            self.info_frame, 
            text=f"Grid: {self.grid_size}x{self.grid_size}",
            font=('Arial', 12)
        )
        self.grid_label.pack(side=tk.RIGHT, padx=20)
        
        # Game grid
        self.grid_frame = tk.Frame(self.main_frame)
        self.grid_frame.pack(pady=20)
        
        # Word input
        self.input_frame = tk.Frame(self.main_frame)
        self.input_frame.pack(pady=10)
        
        self.word_entry = tk.Entry(
            self.input_frame,
            font=('Arial', 16),
            width=20,
            justify='center'
        )
        self.word_entry.pack(side=tk.LEFT, padx=5)
        self.word_entry.bind('<Return>', lambda e: self.check_word())
        
        self.submit_btn = tk.Button(
            self.input_frame,
            text="Submit",
            command=self.check_word,
            font=('Arial', 12),
            bg='lightgreen'
        )
        self.submit_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = tk.Button(
            self.input_frame,
            text="Clear",
            command=self.clear_input,
            font=('Arial', 12),
            bg='lightcoral'
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Word list
        self.words_frame = tk.Frame(self.main_frame)
        self.words_frame.pack(fill=tk.BOTH, expand=True)
        
        self.words_label = tk.Label(
            self.words_frame,
            text="Words to find:",
            font=('Arial', 12, 'underline')
        )
        self.words_label.pack(anchor='w')
        
        self.words_list = tk.Listbox(
            self.words_frame,
            font=('Arial', 12),
            height=6
        )
        self.words_list.pack(fill=tk.BOTH, expand=True)
        
        # Buttons
        self.btn_frame = tk.Frame(self.main_frame)
        self.btn_frame.pack(pady=10)
        
        self.hint_btn = tk.Button(
            self.btn_frame,
            text="Hint",
            command=self.give_hint,
            font=('Arial', 12),
            bg='lightblue'
        )
        self.hint_btn.pack(side=tk.LEFT, padx=10)
        
        self.new_game_btn = tk.Button(
            self.btn_frame,
            text="New Game",
            command=self.new_game,
            font=('Arial', 12),
            bg='lightyellow'
        )
        self.new_game_btn.pack(side=tk.LEFT, padx=10)
        
        # Admin button (for adding words)
        self.admin_btn = tk.Button(
            self.btn_frame,
            text="Add Word",
            command=self.show_add_word_dialog,
            font=('Arial', 12),
            bg='pink'
        )
        self.admin_btn.pack(side=tk.LEFT, padx=10)
    
    def create_grid(self):
        # Clear existing grid
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        
        # Create new grid
        self.cells = []
        for i in range(self.grid_size):
            row = []
            for j in range(self.grid_size):
                cell = tk.Label(
                    self.grid_frame,
                    text=self.current_grid.grid[i][j],
                    font=('Arial', 18, 'bold'),
                    width=3,
                    height=1,
                    relief='ridge',
                    bg='white'
                )
                cell.grid(row=i, column=j, padx=2, pady=2)
                cell.bind('<Button-1>', lambda e, i=i, j=j: self.select_cell(i, j))
                row.append(cell)
            self.cells.append(row)
    
    def select_cell(self, row, col):
        current_text = self.word_entry.get()
        self.word_entry.delete(0, tk.END)
        self.word_entry.insert(0, current_text + self.current_grid.grid[row][col])
    
    def clear_input(self):
        self.word_entry.delete(0, tk.END)
    
    def new_game(self):
        self.level = 1
        self.score = 0
        self.grid_size = INITIAL_GRID_SIZE
        self.start_level()
    
    def start_level(self):
        try:
            self.current_grid = WordGrid(self.grid_size, self.level, self.word_bank)
            self.update_display()
            self.create_grid()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create grid: {str(e)}")
            self.root.quit()
    
    def update_display(self):
        self.level_label.config(text=f"Level: {self.level}/{MAX_LEVELS}")
        self.score_label.config(text=f"Score: {self.score}")
        self.grid_label.config(text=f"Grid: {self.grid_size}x{self.grid_size}")
        
        # Update word list
        self.words_list.delete(0, tk.END)
        for word in self.current_grid.placed_words:
            self.words_list.insert(tk.END, word)
    
    def check_word(self):
        guess = self.word_entry.get().strip().upper()
        self.clear_input()
        
        if not guess:
            return
        
        if guess in self.current_grid.placed_words:
            # Mark word as found
            self.current_grid.placed_words.remove(guess)
            self.score += len(guess) * 10
            self.update_display()
            
            # Highlight the found word
            self.highlight_word(guess)
            
            messagebox.showinfo("Correct!", f"You found: {guess}")
            
            if not self.current_grid.placed_words:
                self.level_complete()
        else:
            messagebox.showerror("Incorrect", f"{guess} is not one of the words")
            self.score = max(0, self.score - 5)
            self.update_display()
    
    def highlight_word(self, word):
        # Reset all cells first
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.cells[i][j].config(bg='white')
        
        # Search for the word in all directions
        directions = [
            (0, 1),   # Right
            (1, 0),    # Down
            (1, 1),    # Down-right
            (1, -1)    # Down-left
        ]
        
        word_len = len(word)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for di, dj in directions:
                    try:
                        if all(self.current_grid.grid[i + k*di][j + k*dj] == word[k] 
                              for k in range(word_len)):
                            # Highlight the word
                            for k in range(word_len):
                                self.cells[i + k*di][j + k*dj].config(bg='lightgreen')
                            return
                    except IndexError:
                        continue
    
    def level_complete(self):
        bonus = self.grid_size * 20
        self.score += bonus
        
        if self.level >= MAX_LEVELS:
            messagebox.showinfo(
                "Game Complete!", 
                f"Congratulations! You completed all levels!\nFinal Score: {self.score}"
            )
            self.new_game()
            return
        
        # Increase grid size every 3 levels
        if self.level % 3 == 0 and self.grid_size < MAX_GRID_SIZE:
            self.grid_size += GRID_INCREMENT
        
        self.level += 1
        messagebox.showinfo(
            "Level Complete!", 
            f"Level {self.level-1} complete!\nBonus: +{bonus} points"
        )
        self.start_level()
    
    def give_hint(self):
        if not self.current_grid.placed_words:
            messagebox.showinfo("Hint", "No words left to find!")
            return
        
        word = random.choice(self.current_grid.placed_words)
        hint = f"Look for a {len(word)}-letter word starting with '{word[0]}'"
        
        # For higher levels, provide more challenging hints
        if self.level > 5:
            similar_words = self.word_bank.get_similar_words(word)
            if similar_words:
                hint += f"\nSimilar words: {', '.join(similar_words[:2])}"
        
        messagebox.showinfo("Hint", hint)
    
    def show_add_word_dialog(self):
        """Show a dialog to add new words to the word bank"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New Word")
        dialog.geometry("400x300")
        
        tk.Label(dialog, text="Word:", font=('Arial', 12)).pack(pady=5)
        word_entry = tk.Entry(dialog, font=('Arial', 12))
        word_entry.pack(pady=5)
        
        tk.Label(dialog, text="Category:", font=('Arial', 12)).pack(pady=5)
        category_var = tk.StringVar(dialog)
        categories = list(self.word_bank.word_data.keys())
        category_var.set(categories[0] if categories else "")
        
        category_menu = tk.OptionMenu(dialog, category_var, *categories)
        category_menu.pack(pady=5)
        
        new_category_entry = tk.Entry(dialog, font=('Arial', 12))
        new_category_entry.pack(pady=5)
        tk.Label(dialog, text="Or enter new category:", font=('Arial', 10)).pack()
        
        result_label = tk.Label(dialog, text="", fg='green')
        result_label.pack(pady=10)
        
        def add_word():
            word = word_entry.get().strip().upper()
            category = category_var.get()
            new_category = new_category_entry.get().strip()
            
            if not word:
                result_label.config(text="Please enter a word", fg='red')
                return
            
            if new_category:
                category = new_category
            
            if not category:
                result_label.config(text="Please select or enter a category", fg='red')
                return
            
            if self.word_bank.add_word(word, category):
                result_label.config(text=f"Added '{word}' to '{category}'", fg='green')
                word_entry.delete(0, tk.END)
                new_category_entry.delete(0, tk.END)
            else:
                result_label.config(text=f"'{word}' already exists in '{category}'", fg='red')
        
        tk.Button(
            dialog,
            text="Add Word",
            command=add_word,
            font=('Arial', 12),
            bg='lightgreen'
        ).pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    game = WordHuntGame(root)
    root.mainloop()