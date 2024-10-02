import pygame
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Initialize Pygame
pygame.init()

# Screen dimensions and colors
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 700
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
YELLOW = (255, 255, 0)

# Set up the Pygame display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Enhanced Dynamic Hand Cricket Puzzle Game")

# Define constants
GRID_SIZE = 6
CELL_SIZE = min(SCREEN_WIDTH // (GRID_SIZE + 8), SCREEN_HEIGHT // (GRID_SIZE + 4))
FONT = pygame.font.SysFont(None, 40)
SMALL_FONT = pygame.font.SysFont(None, 30)

# Load Hugging Face model
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"Error loading Hugging Face model: {e}")
    text_generator = None

# GAN components
def build_generator():
    model = Sequential([
        layers.Dense(128, input_shape=(100,), activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(GRID_SIZE * GRID_SIZE, activation='tanh'),
        layers.Reshape((GRID_SIZE, GRID_SIZE))
    ])
    return model

def build_discriminator():
    model = Sequential([
        layers.Flatten(input_shape=(GRID_SIZE, GRID_SIZE)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()

# Compile the models
generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

# Define loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

@tf.function
def train_step(real_images, batch_size):
    noise = tf.random.normal([batch_size, 100])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def train_gan(epochs=5, batch_size=32):
    print("Starting GAN training...")
    for epoch in range(epochs):
        real_images = tf.random.normal([batch_size, GRID_SIZE, GRID_SIZE], mean=0.5, stddev=0.5)
        g_loss, d_loss = train_step(real_images, batch_size)
        print(f"Epoch {epoch + 1}/{epochs} - D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
    print("GAN training completed.")
    return True

def generate_grid():
    noise = tf.random.normal([1, 100])
    generated_grid = generator(noise, training=False)
    grid = tf.clip_by_value(generated_grid[0] * 5 + 5, 1, 10).numpy().astype(int).tolist()
    return [[str(num) if random.random() > 0.2 else random.choice(['S', 'R']) for num in row] for row in grid]

def predict_difficulty(score):
    return min(max(score / 50, 0.1), 1.0)

def apply_blur(surface, amount):
    if amount <= 0:
        return surface
    
    scaled = pygame.transform.smoothscale(surface, (surface.get_width() // amount, surface.get_height() // amount))
    return pygame.transform.smoothscale(scaled, surface.get_size())

def draw_grid(grid, hidden_cells=[], highlighted_cells=[], selector=None, blur_amount=0):
    grid_surface = pygame.Surface((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE))
    grid_surface.fill(WHITE)
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if (i, j) in highlighted_cells:
                pygame.draw.rect(grid_surface, RED, rect)
            elif selector == (i, j):
                pygame.draw.rect(grid_surface, BLUE, rect)
            else:
                pygame.draw.rect(grid_surface, GRAY, rect, 1)

            if (i, j) not in hidden_cells:
                text = FONT.render(str(grid[i][j]), True, BLACK)
                grid_surface.blit(text, (j * CELL_SIZE + 10, i * CELL_SIZE + 10))

    if blur_amount > 0:
        grid_surface = apply_blur(grid_surface, blur_amount)

    screen.blit(grid_surface, (50, 50))

def generate_commentary(game_state, score, difficulty):
    if text_generator is None:
        return "Commentary unavailable"
    prompt = f"In a cricket game, the current score is {score} and the difficulty level is {difficulty:.2f}. Generate a brief, engaging commentary:"
    try:
        response = text_generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        return response.split(":")[-1].strip()
    except Exception as e:
        print(f"Error generating commentary: {e}")
        return "Exciting game in progress!"

def adjust_difficulty(current_difficulty, user_performance, ai_performance):
    if text_generator is None:
        return min(max(current_difficulty + (ai_performance - user_performance) * 0.1, 0.1), 1.0)
    prompt = f"Given the current difficulty of {current_difficulty:.2f}, user performance of {user_performance}, and AI performance of {ai_performance}, suggest a new difficulty level between 0.1 and 1.0:"
    try:
        response = text_generator(prompt, max_length=20, num_return_sequences=1)[0]['generated_text']
        new_difficulty = float(response.split(":")[-1].strip())
        return max(0.1, min(1.0, new_difficulty))
    except Exception as e:
        print(f"Error adjusting difficulty: {e}")
        return current_difficulty

def draw_ai_choices(choices):
    circle_radius = 25
    spacing = 15
    start_x = SCREEN_WIDTH - (len(choices) * (circle_radius * 2 + spacing))
    start_y = 50

    for i, choice in enumerate(choices):
        x = start_x + i * (circle_radius * 2 + spacing)
        pygame.draw.circle(screen, YELLOW, (x, start_y), circle_radius)
        text = FONT.render(str(choice), True, BLACK)
        text_rect = text.get_rect(center=(x, start_y))
        screen.blit(text, text_rect)

def play_innings(batting_player, target=None):
    grid = generate_grid()
    hidden_cells = []
    score = 0
    difficulty = 0.5
    user_performance = 0
    ai_performance = 0
    
    clock = pygame.time.Clock()
    display_time = 3000
    blur_time = 2000
    
    while True:
        screen.fill(WHITE)
        draw_grid(grid, hidden_cells)
        score_text = FONT.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (SCREEN_WIDTH - 250, 20))
        if target:
            target_text = FONT.render(f"Target: {target}", True, BLACK)
            screen.blit(target_text, (SCREEN_WIDTH - 250, 60))
        
        try:
            commentary = generate_commentary("batting" if batting_player == "User" else "bowling", score, difficulty)
            commentary_text = SMALL_FONT.render(commentary, True, BLUE)
            screen.blit(commentary_text, (50, SCREEN_HEIGHT - 50))
        except Exception as e:
            print(f"Error generating commentary: {e}")
        
        pygame.display.flip()
        
        # Display numbers and apply blur effect
        start_time = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_time < display_time + blur_time:
            current_time = pygame.time.get_ticks() - start_time
            if current_time < display_time:
                blur_amount = 0
            else:
                blur_amount = int((current_time - display_time) / (blur_time / 10)) + 1
            
            screen.fill(WHITE)
            draw_grid(grid, hidden_cells, blur_amount=blur_amount)
            screen.blit(score_text, (SCREEN_WIDTH - 250, 20))
            if target:
                screen.blit(target_text, (SCREEN_WIDTH - 250, 60))
            if 'commentary_text' in locals():
                screen.blit(commentary_text, (50, SCREEN_HEIGHT - 50))
            pygame.display.flip()
            clock.tick(60)
        
        # Generate and display AI choices after blur effect
        ai_choices = generate_ai_choices(difficulty, batting_player == "AI")
        draw_ai_choices(ai_choices)
        pygame.display.flip()
        
        # Wait for user input
        waiting_for_input = True
        while waiting_for_input:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    col = (x - 50) // CELL_SIZE
                    row = (y - 50) // CELL_SIZE
                    if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE and (row, col) not in hidden_cells:
                        cell_value = grid[row][col]
                        if str(cell_value).isdigit():
                            user_choice = int(cell_value)
                            ai_choice = random.choice(ai_choices)
                            if batting_player == "User":
                                display_message(f"User chose {user_choice}, AI chose {ai_choice}")
                                if user_choice == ai_choice:
                                    display_message("Out!")
                                    return score
                                else:
                                    score += user_choice
                                    user_performance += user_choice
                            else:  # AI is batting
                                display_message(f"AI chose {ai_choice}, User chose {user_choice}")
                                if ai_choice == user_choice:
                                    display_message("Out!")
                                    return score
                                else:
                                    score += ai_choice
                                    ai_performance += ai_choice
                            if target and score > target:
                                return score
                            hidden_cells.append((row, col))
                        elif cell_value == 'R':
                            grid = generate_grid()
                            hidden_cells = []
                            display_message("Board reshuffled!")
                        elif cell_value == 'S':
                            flat_grid = [item for sublist in grid for item in sublist]
                            random.shuffle(flat_grid)
                            grid = [flat_grid[i:i + GRID_SIZE] for i in range(0, GRID_SIZE * GRID_SIZE, GRID_SIZE)]
                            display_message("Board shuffled!")
                        waiting_for_input = False
            clock.tick(60)
        
        # Update difficulty based on performance
        difficulty = adjust_difficulty(difficulty, user_performance, ai_performance)
        display_time = max(1000, 3000 - int(difficulty * 2000))
        blur_time = max(500, 2000 - int(difficulty * 1500))

def display_message(message, duration=2000):
    screen.fill(WHITE)
    text = FONT.render(message, True, BLACK)
    screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2 - text.get_height() // 2))
    pygame.display.flip()
    pygame.time.wait(duration)

def get_user_input(prompt):
    input_text = ""
    input_rect = pygame.Rect(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2, SCREEN_WIDTH // 2, 50)
    active = True
    while active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return input_text
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    input_text += event.unicode
        
        screen.fill(WHITE)
        text = FONT.render(prompt, True, BLACK)
        screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2 - 100))
        pygame.draw.rect(screen, BLACK, input_rect, 2)
        text_surface = FONT.render(input_text, True, BLACK)
        screen.blit(text_surface, (input_rect.x + 5, input_rect.y + 5))
        pygame.display.flip()

def toss():
    display_message("Toss time!")
    user_choice = get_user_input("Choose 'odd' or 'even': ")
    if user_choice is None:
        return None, None
    user_number = int(get_user_input("Enter a number between 1 and 10: "))
    if user_number is None:
        return None, None
    ai_number = random.randint(1, 10)
    toss_sum = user_number + ai_number

    display_message(f"AI chose {ai_number}. Total is {toss_sum}.")
    if (toss_sum % 2 == 0 and user_choice.lower() == "even") or (toss_sum % 2 == 1 and user_choice.lower() == "odd"):
        display_message("User wins the toss!")
        decision = get_user_input("Choose to bat or bowl: ")
        if decision is None:
            return None, None
        return "User", decision.lower()
    else:
        display_message("AI wins the toss!")
        decision = random.choice(["bat", "bowl"])
        display_message(f"AI chooses to {decision} first.")
        return "AI", decision

def generate_ai_choices(difficulty, ai_batting):
    base_choices = 3 if ai_batting else 5
    max_choices = 10
    num_choices = min(max_choices, base_choices + int(difficulty * (max_choices - base_choices)))
    return random.sample(range(1, 11), num_choices)

def game_loop():
    overall_difficulty = 0.5
    user_total_score = 0
    ai_total_score = 0
    
    while True:
        toss_result = toss()
        if toss_result is None:
            break
        
        toss_winner, toss_choice = toss_result
        first_innings_player = toss_winner if toss_choice == "bat" else ("AI" if toss_winner == "User" else "User")
        second_innings_player = "AI" if first_innings_player == "User" else "User"
        
        display_message(f"{first_innings_player} will bat first")
        first_innings_score = play_innings(first_innings_player)
        if first_innings_score is None:
            break
        
        display_message(f"First innings over! Score: {first_innings_score}")
        display_message(f"{second_innings_player} needs {first_innings_score + 1} to win")
        
        second_innings_score = play_innings(second_innings_player, first_innings_score + 1)
        if second_innings_score is None:
            break
        
        if second_innings_score > first_innings_score:
            winner = second_innings_player
        else:
            winner = first_innings_player
        
        if winner == "User":
            user_total_score += 1
        else:
            ai_total_score += 1
        
        display_message(f"Game over! {winner} wins!")
        display_message(f"Final Scores: {first_innings_player} - {first_innings_score}, {second_innings_player} - {second_innings_score}")
        display_message(f"Overall Score: User {user_total_score} - {ai_total_score} AI")
        
        overall_difficulty = adjust_difficulty(overall_difficulty, user_total_score, ai_total_score)
        
        play_again = get_user_input("Play again? (y/n): ")
        if play_again is None or play_again.lower() != 'y':
            break
    
    pygame.quit()

def main():
    if train_gan(epochs=30, batch_size=32):
        game_loop()

if __name__ == "__main__":
    main()