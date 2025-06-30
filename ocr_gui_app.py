import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import threading
import warnings
import json
import traceback
warnings.filterwarnings('ignore')

class OCR_GUI_App:
    def __init__(self, root):
        self.root = root
        self.root.title("G5 ANN Training - Optical Character Recognition")
        
        # Get screen dimensions and adjust window size to avoid taskbar overlap
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate window size (90% of screen, leaving space for taskbar)
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.85)
        
        # Center the window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2 - 20  # Slightly higher to avoid taskbar
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(1200, 700)  # Minimum size
        
        # Initialize variables
        self.model = None
        self.label_encoder = None
        self.grid_data = np.zeros((7, 5), dtype=int)
        self.training_data = None
        self.training_labels = None
        
        # Training parameters (default values)
        self.epochs = tk.IntVar(value=100)
        self.batch_size = tk.IntVar(value=8)
        self.learning_rate = tk.DoubleVar(value=0.001)
        self.validation_split = tk.DoubleVar(value=0.2)
        self.dropout_rate1 = tk.DoubleVar(value=0.2)
        self.dropout_rate2 = tk.DoubleVar(value=0.2)
        self.hidden1_neurons = tk.IntVar(value=64)
        self.hidden2_neurons = tk.IntVar(value=32)
        
        # Character set selection variables
        self.include_numbers = tk.BooleanVar(value=True)
        self.include_uppercase = tk.BooleanVar(value=True)
        self.include_lowercase = tk.BooleanVar(value=True)
        self.include_special = tk.BooleanVar(value=True)
        
        # Setup GUI
        self.setup_gui()
        self.create_sample_data()
        
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Create main frames
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls and drawing
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Right panel for visualization
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_left_panel()
        self.setup_right_panel()
        
    def setup_left_panel(self):
        """Setup the left control panel"""
        # Title
        ttk.Label(self.left_panel, text="G5 ANN Training - OCR", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Character selection section
        self.setup_character_selection_section()
        
        # Training section
        self.setup_training_section()
        
        # Drawing section
        self.setup_drawing_section()
    
    def setup_character_selection_section(self):
        """Setup character set selection section"""
        char_frame = ttk.LabelFrame(self.left_panel, text="Character Set Selection", padding=10)
        char_frame.pack(fill=tk.X, pady=5)
        
        # Create a 2x2 grid for checkboxes
        checkbox_frame = ttk.Frame(char_frame)
        checkbox_frame.pack()
        
        # Numbers checkbox
        ttk.Checkbutton(checkbox_frame, text="Numbers (0-9)", 
                       variable=self.include_numbers, command=self.update_character_selection).grid(row=0, column=0, sticky='w', padx=5, pady=2)
        
        # Uppercase checkbox  
        ttk.Checkbutton(checkbox_frame, text="Uppercase (A-Z)", 
                       variable=self.include_uppercase, command=self.update_character_selection).grid(row=0, column=1, sticky='w', padx=5, pady=2)
        
        # Lowercase checkbox
        ttk.Checkbutton(checkbox_frame, text="Lowercase (a-z)", 
                       variable=self.include_lowercase, command=self.update_character_selection).grid(row=1, column=0, sticky='w', padx=5, pady=2)
        
        # Special characters checkbox
        ttk.Checkbutton(checkbox_frame, text="Special (!@#$%)", 
                       variable=self.include_special, command=self.update_character_selection).grid(row=1, column=1, sticky='w', padx=5, pady=2)
        
        # Character count label
        self.char_count_label = ttk.Label(char_frame, text="Selected: 87+ characters", 
                                         font=("Arial", 9), foreground="blue")
        self.char_count_label.pack(pady=5)
        
    def update_character_selection(self):
        """Update character selection and count"""
        count = 0
        
        if self.include_numbers.get():
            count += 10
        if self.include_uppercase.get():
            count += 26
        if self.include_lowercase.get():
            count += 26
        if self.include_special.get():
            count += 25
            
        if count == 0:
            self.char_count_label.config(text="No characters selected!", foreground="red")
        else:
            self.char_count_label.config(text=f"Selected: {count} characters", 
                                        foreground="blue")
        
        # Update training data when character selection changes
        if hasattr(self, 'training_data') and self.training_data is not None:
            # Update training data when character selection changes
            self.create_sample_data()  # Recreate training data with new selection
            if hasattr(self, 'model') and self.model is not None:
                self.train_status.config(text="Model: Needs Retraining", foreground="orange")
        
    def setup_training_section(self):
        """Setup the training control section"""
        train_frame = ttk.LabelFrame(self.left_panel, text="Training", padding=10)
        train_frame.pack(fill=tk.X, pady=5)
        
        # Training status
        self.train_status = ttk.Label(train_frame, text="Model: Not Trained", foreground="red")
        self.train_status.pack()
        
        # Train button
        self.train_btn = ttk.Button(train_frame, text="Train Model", command=self.train_model)
        self.train_btn.pack(pady=5)
        
        # Training progress
        self.progress = ttk.Progressbar(train_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
    def setup_drawing_section(self):
        """Setup the drawing canvas section"""
        draw_frame = ttk.LabelFrame(self.left_panel, text="Draw Character (5x7)", padding=10)
        draw_frame.pack(fill=tk.X, pady=5)
        
        # Drawing canvas
        self.canvas = tk.Canvas(draw_frame, width=250, height=350, bg="white", relief=tk.SUNKEN, bd=2)
        self.canvas.pack()
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.toggle_cell)
        self.canvas.bind("<B1-Motion>", self.toggle_cell)
        
        # Control buttons
        btn_frame = ttk.Frame(draw_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Clear", command=self.clear_canvas).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Recognize", command=self.recognize_drawn).pack(side=tk.RIGHT, padx=2)
        
        # Draw initial grid
        self.draw_grid()
    
    def log_message(self, message):
        """Log a message to the console with timestamp"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def setup_recognition_tab(self):
        """Setup the recognition results tab"""
        # Create main container with padding
        main_container = ttk.Frame(self.results_tab)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        ttk.Label(main_container, text="Character Recognition Results", 
                 font=("Arial", 18, "bold")).pack(pady=(0, 20))
        
        # Current Recognition Section
        current_frame = ttk.LabelFrame(main_container, text="Current Recognition", padding=20)
        current_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Results display in a grid
        results_grid = ttk.Frame(current_frame)
        results_grid.pack(fill=tk.X, expand=True)
        
        # ASCII code - highlighted at top as requested
        ascii_container = ttk.Frame(results_grid)
        ascii_container.pack(fill=tk.X, pady=10)
        
        ttk.Label(ascii_container, text="ASCII Code:", 
                 font=("Arial", 16, "bold")).pack()
        self.ascii_result = ttk.Label(ascii_container, text="-", 
                                     font=("Arial", 36, "bold"), 
                                     foreground="blue")
        self.ascii_result.pack(pady=5)
        
        # Character result - at bottom as requested
        char_container = ttk.Frame(results_grid)
        char_container.pack(fill=tk.X, pady=10)
        
        ttk.Label(char_container, text="Recognized Character:", 
                 font=("Arial", 14)).pack()
        self.char_result = ttk.Label(char_container, text="-", 
                                    font=("Arial", 48, "bold"),
                                    foreground="darkgreen")
        self.char_result.pack(pady=5)
        
        # Confidence score
        conf_container = ttk.Frame(results_grid)
        conf_container.pack(fill=tk.X, pady=10)
        
        ttk.Label(conf_container, text="Confidence Score:", 
                 font=("Arial", 14)).pack()
        self.conf_result = ttk.Label(conf_container, text="-", 
                                    font=("Arial", 16))
        self.conf_result.pack(pady=5)
        
        # Recognition Instructions
        instructions_frame = ttk.LabelFrame(main_container, text="Instructions", padding=15)
        instructions_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(instructions_frame, text="""1. Select the desired character sets in the left panel
2. Train the model using the Training Parameters tab
3. Draw a character in the 5x7 grid on the left panel
4. Click 'Recognize' to see results here
5. Results will only include characters from your selected sets""", 
                 font=("Arial", 11), justify=tk.LEFT).pack()
        
        # Status section
        status_frame = ttk.LabelFrame(main_container, text="Recognition Status", padding=15)
        status_frame.pack(fill=tk.X)
        
        self.recognition_status = ttk.Label(status_frame, text="Ready for recognition", 
                                          font=("Arial", 12), foreground="green")
        self.recognition_status.pack()
        
    def setup_right_panel(self):
        """Setup the right visualization panel"""
        # Create main container for right panel layout
        right_container = ttk.Frame(self.right_panel)
        right_container.pack(fill=tk.BOTH, expand=True)
        
        # Top section for tabs
        tabs_frame = ttk.Frame(right_container)
        tabs_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(tabs_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Training parameters tab
        self.params_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.params_tab, text="Training Parameters")
        
        # Training visualization tab
        self.train_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text="Training Progress")
        
        # Network visualization tab
        self.network_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.network_tab, text="Network Processing")
        
        # Recognition results tab
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Recognition Results")
        
        # ASCII table tab
        self.ascii_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.ascii_tab, text="ASCII Table")
        
        # Setup all components
        self.setup_parameters_tab()
        self.setup_training_plot()
        self.setup_network_plot()
        self.setup_recognition_tab()
        self.setup_ascii_table_tab()
    
    def setup_parameters_tab(self):
        """Setup the training parameters configuration tab"""
        # Create scrollable frame for parameters
        canvas = tk.Canvas(self.params_tab)
        scrollbar = ttk.Scrollbar(self.params_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Title
        title_frame = ttk.Frame(scrollable_frame)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(title_frame, text="Neural Network Training Parameters", 
                 font=("Arial", 14, "bold")).pack()
        
        # Training Parameters Section
        train_params_frame = ttk.LabelFrame(scrollable_frame, text="Training Parameters", padding=10)
        train_params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Epochs
        epochs_frame = ttk.Frame(train_params_frame)
        epochs_frame.pack(fill=tk.X, pady=2)
        ttk.Label(epochs_frame, text="Epochs:", width=20).pack(side=tk.LEFT)
        epochs_spinbox = ttk.Spinbox(epochs_frame, from_=10, to=1000, width=10, textvariable=self.epochs)
        epochs_spinbox.pack(side=tk.LEFT)
        ttk.Label(epochs_frame, text="(Number of training iterations)").pack(side=tk.LEFT, padx=(10,0))
        
        # Batch Size
        batch_frame = ttk.Frame(train_params_frame)
        batch_frame.pack(fill=tk.X, pady=2)
        ttk.Label(batch_frame, text="Batch Size:", width=20).pack(side=tk.LEFT)
        batch_spinbox = ttk.Spinbox(batch_frame, from_=1, to=64, width=10, textvariable=self.batch_size)
        batch_spinbox.pack(side=tk.LEFT)
        ttk.Label(batch_frame, text="(Samples processed per update)").pack(side=tk.LEFT, padx=(10,0))
        
        # Learning Rate
        lr_frame = ttk.Frame(train_params_frame)
        lr_frame.pack(fill=tk.X, pady=2)
        ttk.Label(lr_frame, text="Learning Rate:", width=20).pack(side=tk.LEFT)
        lr_spinbox = ttk.Spinbox(lr_frame, from_=0.0001, to=0.1, increment=0.0001, width=10, textvariable=self.learning_rate)
        lr_spinbox.pack(side=tk.LEFT)
        ttk.Label(lr_frame, text="(Step size for weight updates)").pack(side=tk.LEFT, padx=(10,0))
        
        # Validation Split
        val_frame = ttk.Frame(train_params_frame)
        val_frame.pack(fill=tk.X, pady=2)
        ttk.Label(val_frame, text="Validation Split:", width=20).pack(side=tk.LEFT)
        val_spinbox = ttk.Spinbox(val_frame, from_=0.1, to=0.5, increment=0.05, width=10, textvariable=self.validation_split)
        val_spinbox.pack(side=tk.LEFT)
        ttk.Label(val_frame, text="(Fraction of data for validation)").pack(side=tk.LEFT, padx=(10,0))
        
        # Network Architecture Section
        arch_frame = ttk.LabelFrame(scrollable_frame, text="Network Architecture", padding=10)
        arch_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Hidden Layer 1
        h1_frame = ttk.Frame(arch_frame)
        h1_frame.pack(fill=tk.X, pady=2)
        ttk.Label(h1_frame, text="Hidden Layer 1:", width=20).pack(side=tk.LEFT)
        h1_spinbox = ttk.Spinbox(h1_frame, from_=32, to=256, width=10, textvariable=self.hidden1_neurons)
        h1_spinbox.pack(side=tk.LEFT)
        ttk.Label(h1_frame, text="neurons").pack(side=tk.LEFT, padx=(5,0))
        
        # Hidden Layer 2
        h2_frame = ttk.Frame(arch_frame)
        h2_frame.pack(fill=tk.X, pady=2)
        ttk.Label(h2_frame, text="Hidden Layer 2:", width=20).pack(side=tk.LEFT)
        h2_spinbox = ttk.Spinbox(h2_frame, from_=32, to=128, width=10, textvariable=self.hidden2_neurons)
        h2_spinbox.pack(side=tk.LEFT)
        ttk.Label(h2_frame, text="neurons").pack(side=tk.LEFT, padx=(5,0))
        
        # Regularization Section
        reg_frame = ttk.LabelFrame(scrollable_frame, text="Regularization (Dropout)", padding=10)
        reg_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Dropout Rate 1
        dr1_frame = ttk.Frame(reg_frame)
        dr1_frame.pack(fill=tk.X, pady=2)
        ttk.Label(dr1_frame, text="Dropout Rate 1:", width=20).pack(side=tk.LEFT)
        dr1_spinbox = ttk.Spinbox(dr1_frame, from_=0.0, to=0.8, increment=0.05, width=10, textvariable=self.dropout_rate1)
        dr1_spinbox.pack(side=tk.LEFT)
        ttk.Label(dr1_frame, text="(After Hidden Layer 1)").pack(side=tk.LEFT, padx=(10,0))
        
        # Dropout Rate 2
        dr2_frame = ttk.Frame(reg_frame)
        dr2_frame.pack(fill=tk.X, pady=2)
        ttk.Label(dr2_frame, text="Dropout Rate 2:", width=20).pack(side=tk.LEFT)
        dr2_spinbox = ttk.Spinbox(dr2_frame, from_=0.0, to=0.8, increment=0.05, width=10, textvariable=self.dropout_rate2)
        dr2_spinbox.pack(side=tk.LEFT)
        ttk.Label(dr2_frame, text="(After Hidden Layer 2)").pack(side=tk.LEFT, padx=(10,0))
        
        # Buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=20)
        
        ttk.Button(button_frame, text="Reset to Defaults", 
                  command=self.reset_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Parameters", 
                  command=self.save_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Parameters", 
                  command=self.load_parameters).pack(side=tk.LEFT, padx=5)
        
        # Information section
        info_frame = ttk.LabelFrame(scrollable_frame, text="Information", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(info_frame, text="""
Network Architecture (Optimized for Overfitting Prevention):
Input Layer: 35 neurons (5x7 matrix)
Hidden Layer 1: Configurable (default: 64 neurons, ReLU) + Dropout
Hidden Layer 2: Configurable (default: 32 neurons, ReLU) + Dropout
Output Layer: Variable neurons (number of selected characters)

Anti-Overfitting Features:
• Double dropout layers after both hidden layers
• Early stopping with patience=10 epochs
• Validation-based weight restoration
• Optimal for larger character sets (4 sets selected)
• Prevents overfitting while maintaining accuracy
        """, justify=tk.LEFT, font=("Arial", 9)).pack()
        
    def reset_parameters(self):
        """Reset parameters to default values"""
        self.epochs.set(100)
        self.batch_size.set(8)
        self.learning_rate.set(0.001)
        self.validation_split.set(0.2)
        self.dropout_rate1.set(0.2)
        self.dropout_rate2.set(0.2)
        self.hidden1_neurons.set(64)
        self.hidden2_neurons.set(32)
        self.include_numbers.set(True)
        self.include_uppercase.set(True)
        self.include_lowercase.set(True)
        self.include_special.set(True)
        
        # Update character selection display
        self.update_character_selection()
        
    def save_parameters(self):
        """Save parameters to file"""
        params = {
            'epochs': self.epochs.get(),
            'batch_size': self.batch_size.get(),
            'learning_rate': self.learning_rate.get(),
            'validation_split': self.validation_split.get(),
            'dropout_rate1': self.dropout_rate1.get(),
            'dropout_rate2': self.dropout_rate2.get(),
            'hidden1_neurons': self.hidden1_neurons.get(),
            'hidden2_neurons': self.hidden2_neurons.get(),
            'include_numbers': self.include_numbers.get(),
            'include_uppercase': self.include_uppercase.get(),
            'include_lowercase': self.include_lowercase.get(),
            'include_special': self.include_special.get()
        }
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save Training Parameters"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(params, f, indent=2)
                messagebox.showinfo("Success", f"Parameters saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving parameters: {e}")
    
    def load_parameters(self):
        """Load parameters from file"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Load Training Parameters"
            )
            
            if filename:
                with open(filename, 'r') as f:
                    params = json.load(f)
                
                # Set parameters
                self.epochs.set(params.get('epochs', 100))
                self.batch_size.set(params.get('batch_size', 8))
                self.learning_rate.set(params.get('learning_rate', 0.001))
                self.validation_split.set(params.get('validation_split', 0.2))
                self.dropout_rate1.set(params.get('dropout_rate1', 0.2))
                self.dropout_rate2.set(params.get('dropout_rate2', 0.2))
                self.hidden1_neurons.set(params.get('hidden1_neurons', 64))
                self.hidden2_neurons.set(params.get('hidden2_neurons', 32))
                self.include_numbers.set(params.get('include_numbers', True))
                self.include_uppercase.set(params.get('include_uppercase', True))
                self.include_lowercase.set(params.get('include_lowercase', True))
                self.include_special.set(params.get('include_special', True))
                
                # Update character selection display
                self.update_character_selection()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading parameters: {e}")
        
    def setup_training_plot(self):
        """Setup the training progress plot"""
        self.train_fig = Figure(figsize=(8, 6), dpi=100)
        self.train_canvas = FigureCanvasTkAgg(self.train_fig, self.train_tab)
        self.train_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial empty plot
        ax = self.train_fig.add_subplot(111)
        ax.set_title("Training Progress (Train Model First)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy/Loss")
        self.train_fig.tight_layout()
        
    def setup_network_plot(self):
        """Setup the network processing visualization"""
        self.network_fig = Figure(figsize=(12, 8), dpi=100)
        self.network_canvas = FigureCanvasTkAgg(self.network_fig, self.network_tab)
        self.network_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial empty plot
        ax = self.network_fig.add_subplot(111)
        ax.set_title("Network Processing Visualization (Draw and Recognize First)")
        ax.text(0.5, 0.5, "No data to visualize", ha='center', va='center', transform=ax.transAxes)
        self.network_fig.tight_layout()
        
    def draw_grid(self):
        """Draw the 5x7 grid on canvas"""
        self.canvas.delete("all")
        
        cell_width = 40
        cell_height = 40
        start_x = 25
        start_y = 25
        
        for row in range(7):
            for col in range(5):
                x1 = start_x + col * cell_width
                y1 = start_y + row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                # Color cell based on data
                color = "black" if self.grid_data[row, col] == 1 else "white"
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray", tags=f"cell_{row}_{col}")
                
    def toggle_cell(self, event):
        """Toggle cell on mouse click/drag"""
        # Find which cell was clicked
        cell_width = 40
        cell_height = 40
        start_x = 25
        start_y = 25
        
        col = (event.x - start_x) // cell_width
        row = (event.y - start_y) // cell_height
        
        if 0 <= row < 7 and 0 <= col < 5:
            self.grid_data[row, col] = 1 - self.grid_data[row, col]  # Toggle
            self.draw_grid()
            
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.grid_data = np.zeros((7, 5), dtype=int)
        self.draw_grid()
        
    def create_sample_data(self):
        """Create training data based on selected character sets"""
        try:
            from character_patterns import CharacterPatterns
            
            # Create character patterns instance
            char_patterns = CharacterPatterns()
            
            # Filter characters based on selection
            selected_chars = self.get_selected_characters(char_patterns)
            
            if not selected_chars:
                return
            
            # Get training data for selected characters only
            self.training_data, self.training_labels = char_patterns.get_filtered_training_data(
                selected_chars, noise_variations=2)
            

            
        except ImportError:
            # Fallback to basic patterns if character_patterns.py not available
            self.create_basic_patterns()
    
    def get_selected_characters(self, char_patterns):
        """Get list of characters based on selection checkboxes"""
        selected = []
        
        if self.include_numbers.get():
            selected.extend([str(i) for i in range(10)])  # 0-9
        if self.include_uppercase.get():
            selected.extend([chr(i) for i in range(ord('A'), ord('Z')+1)])  # A-Z
        if self.include_lowercase.get():
            selected.extend([chr(i) for i in range(ord('a'), ord('z')+1)])  # a-z
        if self.include_special.get():
            # Add special characters
            special_chars = ['!', '@', '#', '$', '%', '&', '*', '+', '-', '=', '?', '.', ',', ';', ':', "'", '"', '(', ')', '[', ']', '\\', '_', ' ']
            selected.extend(special_chars)
        
        return selected
    
    def create_basic_patterns(self):
        """Fallback method with basic patterns"""
        patterns = {
            '0': [[0,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [0,1,1,1,0]],
            '1': [[0,0,1,0,0], [0,1,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,1,1,1,0]],
            '2': [[0,1,1,1,0], [1,0,0,0,1], [0,0,0,0,1], [0,0,0,1,0], [0,0,1,0,0], [0,1,0,0,0], [1,1,1,1,1]],
            '3': [[0,1,1,1,0], [1,0,0,0,1], [0,0,0,0,1], [0,0,1,1,0], [0,0,0,0,1], [1,0,0,0,1], [0,1,1,1,0]],
            '4': [[0,0,0,1,0], [0,0,1,1,0], [0,1,0,1,0], [1,0,0,1,0], [1,1,1,1,1], [0,0,0,1,0], [0,0,0,1,0]],
            'A': [[0,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1]],
            'B': [[1,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,0]],
            'C': [[0,1,1,1,0], [1,0,0,0,1], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,1], [0,1,1,1,0]]
        }
        
        X = []
        y = []
        
        for char, pattern in patterns.items():
            flattened = np.array(pattern).flatten()
            X.append(flattened)
            y.append(char)
            
            # Add variations
            for _ in range(3):
                noisy_pattern = flattened.copy()
                noise_indices = np.random.choice(35, size=2, replace=False)
                for idx in noise_indices:
                    if np.random.random() > 0.7:
                        noisy_pattern[idx] = 1 - noisy_pattern[idx]
                X.append(noisy_pattern)
                y.append(char)
        
        self.training_data = np.array(X)
        self.training_labels = np.array(y)
        
    def train_model(self):
        """Train the neural network model"""
        if self.training_data is None:
            messagebox.showerror("Error", "No training data available!")
            return
            
        def train_thread():
            try:
                self.progress.start()
                self.train_btn.config(state='disabled')
                
                # Encode labels
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(self.training_labels)
                
                # Build model
                num_classes = len(np.unique(y_encoded))
                self.model = self.build_model(input_size=35, num_classes=num_classes)
                
                # Train/validation split
                val_split = self.validation_split.get()
                if len(self.training_data) > 20:
                    X_train, X_val, y_train, y_val = train_test_split(
                        self.training_data, y_encoded, test_size=val_split, random_state=42
                    )
                else:
                    # For small datasets, use the same data for validation
                    X_train, X_val, y_train, y_val = self.training_data, self.training_data, y_encoded, y_encoded
                
                self.log_message("Starting training...")
                
                # Train model
                batch_size = min(self.batch_size.get(), len(X_train))
                
                # Early stopping to prevent overfitting
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=0
                )
                
                history = self.model.fit(
                    X_train, y_train,
                    epochs=self.epochs.get(),
                    validation_data=(X_val, y_val),
                    verbose=0,
                    batch_size=batch_size,
                    callbacks=[early_stopping]
                )
                
                # Check if early stopping was triggered
                epochs_completed = len(history.history['accuracy'])
                if epochs_completed < self.epochs.get():
                    self.log_message(f"Early stopping triggered after {epochs_completed} epochs (prevented overfitting)")
                else:
                    self.log_message(f"Training completed all {epochs_completed} epochs")
                
                # Final evaluation
                final_acc = history.history['accuracy'][-1]
                final_val_acc = history.history['val_accuracy'][-1]
                
                self.log_message("=== Training Results ===")
                self.log_message(f"Final Training Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
                self.log_message(f"Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
                
                if final_val_acc > 0.9:
                    self.log_message("Excellent performance! Model ready for use.")
                elif final_val_acc > 0.7:
                    self.log_message("Good performance. Consider more training for better accuracy.")
                else:
                    self.log_message("Low accuracy. Try adjusting parameters or more training.")
                
                self.train_status.config(text="Model: Trained", foreground="green")
                
                # Update training plot
                self.update_training_plot(history)
                
            except Exception as e:
                self.log_message(f"Training error: {e}")
                self.log_message(f"Error details: {traceback.format_exc()}")
                self.train_status.config(text="Model: Training Failed", foreground="red")
                messagebox.showerror("Training Error", str(e))
            finally:
                self.progress.stop()
                self.train_btn.config(state='normal')
        
        threading.Thread(target=train_thread, daemon=True).start()
        
    def build_model(self, input_size=35, num_classes=8):
        """Build the neural network model with simplified architecture for small datasets"""
        # Use simplified architecture - only 2 hidden layers for limited training data
        model = keras.Sequential([
            layers.Input(shape=(input_size,)),
            layers.Dense(self.hidden1_neurons.get(), activation='relu', name='hidden_1'),
            layers.Dropout(self.dropout_rate1.get()),
            layers.Dense(self.hidden2_neurons.get(), activation='relu', name='hidden_2'),
            layers.Dropout(self.dropout_rate2.get()),
            layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        # Use configurable learning rate
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate.get())
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def update_training_plot(self, history):
        """Update the training progress visualization"""
        self.train_fig.clear()
        
        # Create subplots
        ax1 = self.train_fig.add_subplot(211)
        ax2 = self.train_fig.add_subplot(212)
        
        # Plot accuracy
        epochs = range(1, len(history.history['accuracy']) + 1)
        ax1.plot(epochs, history.history['accuracy'], 'bo-', label='Training Accuracy', markersize=3)
        ax1.plot(epochs, history.history['val_accuracy'], 'ro-', label='Validation Accuracy', markersize=3)
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(epochs, history.history['loss'], 'bo-', label='Training Loss', markersize=3)
        ax2.plot(epochs, history.history['val_loss'], 'ro-', label='Validation Loss', markersize=3)
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        self.train_fig.tight_layout()
        self.train_canvas.draw()
        
    def recognize_drawn(self):
        """Recognize the drawn character"""
        if self.model is None:
            self.recognition_status.config(text="Error: No trained model available", foreground="red")
            messagebox.showerror("Error", "Please train the model first!")
            return
            
        try:
            # Update status
            self.recognition_status.config(text="Processing recognition...", foreground="orange")
            
            # Get flattened pattern
            pattern = self.grid_data.flatten()
            
            # Make prediction
            prediction = self.model.predict(pattern.reshape(1, -1), verbose=0)
            
            # Get selected characters for filtering
            selected_chars = self.get_selected_characters(None)
            if not selected_chars:
                self.recognition_status.config(text="Error: No character sets selected", foreground="red")
                self.log_message("Error: No character sets selected!")
                return
            
            # Filter predictions to only include selected character sets
            filtered_prediction = np.zeros_like(prediction[0])
            all_classes = self.label_encoder.classes_
            
            for i, char in enumerate(all_classes):
                if char in selected_chars:
                    filtered_prediction[i] = prediction[0][i]
            
            # Renormalize the filtered predictions
            if np.sum(filtered_prediction) > 0:
                filtered_prediction = filtered_prediction / np.sum(filtered_prediction)
            
            # Get the best prediction from filtered results
            predicted_class = np.argmax(filtered_prediction)
            confidence = np.max(filtered_prediction)
            
            # Decode label
            predicted_char = all_classes[predicted_class]
            ascii_code = ord(predicted_char)
            
            # Update results in the new format
            self.char_result.config(text=f"{predicted_char}")
            self.ascii_result.config(text=f"{ascii_code}")
            self.conf_result.config(text=f"{confidence:.4f} ({confidence*100:.2f}%)")
            self.recognition_status.config(text=f"Recognition completed successfully", foreground="green")
            
            # Update network visualization with filtered prediction
            filtered_prediction_reshaped = filtered_prediction.reshape(1, -1)
            self.update_network_visualization(pattern, filtered_prediction_reshaped)
            
            self.log_message(f"Recognition: '{predicted_char}' (ASCII: {ascii_code}, Conf: {confidence:.4f})")
            
        except Exception as e:
            self.log_message(f"Recognition error: {str(e)}")
            self.recognition_status.config(text=f"Recognition error: {str(e)}", foreground="red")
            messagebox.showerror("Recognition Error", str(e))
            
    def update_network_visualization(self, input_pattern, final_prediction):
        """Update the network processing visualization with modern, professional UI/UX"""
        try:
            self.network_fig.clear()
            self.network_fig.patch.set_facecolor('#f8f9fa')  # Light background
            
            # Create a modern card-based layout
            gs = self.network_fig.add_gridspec(4, 5, hspace=0.4, wspace=0.3,
                                             height_ratios=[0.8, 1.5, 0.3, 1.2],
                                             width_ratios=[1, 0.2, 1, 0.2, 1])
            
            # === SECTION 1: INPUT PROCESSING ===
            ax_input = self.network_fig.add_subplot(gs[1, 0])
            input_2d = input_pattern.reshape(7, 5)
            
            # Create a more aesthetic input visualization
            ax_input.imshow(input_2d, cmap='RdBu_r', interpolation='nearest', 
                          vmin=0, vmax=1, aspect='equal')
            ax_input.set_title('INPUT\nCharacter Pattern', fontsize=11, fontweight='bold', 
                             color='#2c3e50', pad=10)
            ax_input.set_xticks([])
            ax_input.set_yticks([])
            
            # Add subtle border and grid
            for spine in ax_input.spines.values():
                spine.set_color('#34495e')
                spine.set_linewidth(2)
            
            # Add grid lines for better visualization
            for i in range(6):
                ax_input.axhline(i-0.5, color='#34495e', linewidth=0.5, alpha=0.4)
            for i in range(4):
                ax_input.axvline(i-0.5, color='#34495e', linewidth=0.5, alpha=0.4)
            
            # === SECTION 2: HIDDEN LAYER 1 ===
            ax_layer1 = self.network_fig.add_subplot(gs[1, 2])
            temp_model1 = keras.Model(inputs=self.model.input, 
                                    outputs=self.model.get_layer('hidden_1').output)
            layer1_output = temp_model1.predict(input_pattern.reshape(1, -1), verbose=0)[0]
            
            # Create modern heatmap visualization
            n1 = len(layer1_output)
            size1 = int(np.ceil(np.sqrt(n1)))
            padded1 = np.zeros(size1 * size1)
            padded1[:n1] = layer1_output
            
            im1 = ax_layer1.imshow(padded1.reshape(size1, size1), cmap='viridis', 
                                 interpolation='nearest', aspect='equal')
            ax_layer1.set_title('LAYER 1\nFeature Detection', fontsize=11, fontweight='bold', 
                              color='#8e44ad', pad=10)
            ax_layer1.set_xticks([])
            ax_layer1.set_yticks([])
            
            # Style the subplot
            for spine in ax_layer1.spines.values():
                spine.set_color('#8e44ad')
                spine.set_linewidth(2)
            
            # === SECTION 3: HIDDEN LAYER 2 ===
            ax_layer2 = self.network_fig.add_subplot(gs[1, 4])
            temp_model2 = keras.Model(inputs=self.model.input, 
                                    outputs=self.model.get_layer('hidden_2').output)
            layer2_output = temp_model2.predict(input_pattern.reshape(1, -1), verbose=0)[0]
            
            n2 = len(layer2_output)
            size2 = int(np.ceil(np.sqrt(n2)))
            padded2 = np.zeros(size2 * size2)
            padded2[:n2] = layer2_output
            
            im2 = ax_layer2.imshow(padded2.reshape(size2, size2), cmap='plasma', 
                                 interpolation='nearest', aspect='equal')
            ax_layer2.set_title('LAYER 2\nPattern Recognition', fontsize=11, fontweight='bold', 
                              color='#e74c3c', pad=10)
            ax_layer2.set_xticks([])
            ax_layer2.set_yticks([])
            
            # Style the subplot
            for spine in ax_layer2.spines.values():
                spine.set_color('#e74c3c')
                spine.set_linewidth(2)
            
            # === FLOW ARROWS (Better positioned) ===
            # Create invisible axes for drawing arrows in middle row
            ax_arrows = self.network_fig.add_subplot(gs[2, :])
            ax_arrows.set_xlim(0, 1)
            ax_arrows.set_ylim(0, 1)
            ax_arrows.axis('off')
            
            
            
            # === PREDICTION RESULTS SECTION ===
            ax_results = self.network_fig.add_subplot(gs[3, :])
            ax_results.axis('off')
            
            classes = self.label_encoder.classes_
            probabilities = final_prediction[0]
            predicted_char = classes[np.argmax(probabilities)]
            confidence = np.max(probabilities)
            
            # Get top 10 predictions
            top_indices = np.argsort(probabilities)[-10:][::-1]
            top_classes = [classes[i] for i in top_indices]
            top_probs = probabilities[top_indices]
            
            # Create modern prediction display
            result_text = f"PREDICTION RESULT:  '{predicted_char}'  •  Confidence: {confidence:.1%}  •  ASCII: {ord(predicted_char)}\n\n TOP 10 PREDICTIONS:\n"
            
            for i, (char, prob) in enumerate(zip(top_classes, top_probs)):
                status = " " if i == 0 else " " if i == 1 else " " if i == 2 else " " if i == 3 else " " if i == 4 else " " if i == 5 else " " if i == 6 else " " if i == 7 else " " if i == 8 else " " if i == 9 else "  "
                result_text += f"   {status} '{char}' → {prob:.1%} (ASCII: {ord(char)})\n"
            
            # Confidence interpretation
            if confidence > 0.8:
                interpretation = "HIGH CONFIDENCE - Clear character recognition"
            elif confidence > 0.5:
                interpretation = "MEDIUM CONFIDENCE - Some uncertainty detected"
            else:
                interpretation = "LOW CONFIDENCE - Ambiguous or unclear pattern"
            
            result_text += f"\nINTERPRETATION: {interpretation}"
            
            # Display with modern styling
            ax_results.text(0.02, 1.5, result_text, transform=ax_results.transAxes,
                          fontsize=12, verticalalignment='top', fontfamily='Arial',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor='#ecf0f1', 
                                  edgecolor='#bdc3c7', alpha=0.95))
            
            # === TECHNICAL INFO SECTION ===
            tech_info = f"NETWORK ARCHITECTURE:  Input: 35 inputs  →  Hidden 1: {self.hidden1_neurons.get()} neurons  →  Hidden 2: {self.hidden2_neurons.get()} neurons  →  Output: {len(classes)} classes"
            
            ax_results.text(0.02, 1.7, tech_info, transform=ax_results.transAxes,
                          fontsize=10, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='#d5dbdb', 
                                  edgecolor='#95a5a6', alpha=0.9))
            
            # === MAIN TITLE ===
            self.network_fig.suptitle('NEURAL NETWORK PROCESSING', 
                                    fontsize=16, fontweight='bold', color='#2c3e50', y=0.95)
            
            
            self.network_canvas.draw()
            
        except Exception as e:
            self.log_message(f"Visualization error: {str(e)}")
            
    def setup_ascii_table_tab(self):
        """Setup the ASCII table tab"""
        # Create main container with padding
        main_container = ttk.Frame(self.ascii_tab)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        ttk.Label(main_container, text="ASCII Character Reference Table", 
                 font=("Arial", 18, "bold")).pack(pady=(0, 20))
        
        # Create notebook for different character categories
        table_notebook = ttk.Notebook(main_container)
        table_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Numbers tab
        numbers_frame = ttk.Frame(table_notebook)
        table_notebook.add(numbers_frame, text="Numbers (0-9)")
        self.create_ascii_table(numbers_frame, [(str(i), ord(str(i))) for i in range(10)])
        
        # Uppercase letters tab
        uppercase_frame = ttk.Frame(table_notebook)
        table_notebook.add(uppercase_frame, text="Uppercase (A-Z)")
        uppercase_chars = [(chr(i), i) for i in range(ord('A'), ord('Z')+1)]
        self.create_ascii_table(uppercase_frame, uppercase_chars)
        
        # Lowercase letters tab
        lowercase_frame = ttk.Frame(table_notebook)
        table_notebook.add(lowercase_frame, text="Lowercase (a-z)")
        lowercase_chars = [(chr(i), i) for i in range(ord('a'), ord('z')+1)]
        self.create_ascii_table(lowercase_frame, lowercase_chars)
        
        # Special characters tab
        special_frame = ttk.Frame(table_notebook)
        table_notebook.add(special_frame, text="Special Characters")
        special_chars = [('!', 33), ('@', 64), ('#', 35), ('$', 36), ('%', 37), ('&', 38), 
                        ('*', 42), ('+', 43), ('-', 45), ('=', 61), ('?', 63), ('.', 46), 
                        (',', 44), (';', 59), (':', 58), ("'", 39), ('"', 34), ('(', 40), 
                        (')', 41), ('[', 91), (']', 93), ('\\', 92), ('_', 95), (' ', 32)]
        self.create_ascii_table(special_frame, special_chars)
        
    def create_ascii_table(self, parent, char_list):
        """Create an ASCII table for the given character list"""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create table headers
        header_frame = ttk.Frame(scrollable_frame)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(header_frame, text="Character", font=("Arial", 14, "bold"), width=20).pack(side=tk.LEFT)
        ttk.Label(header_frame, text="ASCII Code", font=("Arial", 14, "bold"), width=20).pack(side=tk.LEFT)
        
        # Create separator
        ttk.Separator(scrollable_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=5)
        
        # Create table rows
        for char, ascii_val in char_list:
            row_frame = ttk.Frame(scrollable_frame)
            row_frame.pack(fill=tk.X, padx=10, pady=2)
            
            # Character (display space as 'SPACE')
            display_char = 'SPACE' if char == ' ' else char
            ttk.Label(row_frame, text=f"'{display_char}'", font=("Arial", 12), width=20).pack(side=tk.LEFT)
            
            # ASCII code
            ttk.Label(row_frame, text=str(ascii_val), font=("Arial", 12), width=20).pack(side=tk.LEFT)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

def main():
    root = tk.Tk()
    app = OCR_GUI_App(root)
    root.mainloop()

if __name__ == "__main__":
    main() 