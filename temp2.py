import joblib
import random
import time
import torch
import torch.nn as nn
import numpy as np

# ==========================
# QUIZ LOGIC
# ==========================

class SimpleBKT:
    def __init__(self, skills, initial_mastery=0.2, p_transit=0.2, p_slip=0.1, p_guess=0.3):
        """
        Initialize Bayesian Knowledge Tracing (BKT) model.
        :param skills: List of skill names
        :param initial_mastery: Initial probability of mastery per skill (default: 0.2 for all)
        :param p_transit: Probability of learning after each attempt
        :param p_slip: Probability of making a mistake despite mastery
        :param p_guess: Probability of guessing correctly despite not mastering
        """
        self.skills = skills
        self.mastery = {skill: initial_mastery for skill in skills}
        self.p_transit = p_transit
        self.p_slip = p_slip
        self.p_guess = p_guess

    def update(self, skill, correct):
        """
        Update mastery probability after an attempt.
        :param skill: The skill being practiced
        :param correct: Whether the student answered correctly (1) or incorrectly (0)
        """
        if skill not in self.mastery:
            raise ValueError(f"Skill {skill} not found in model.")

        p_mastery = self.mastery[skill]

        if correct:
            # Probability of being correct given mastery
            p_correct = p_mastery * (1 - self.p_slip) + (1 - p_mastery) * self.p_guess
            p_mastery_given_obs = (p_mastery * (1 - self.p_slip)) / p_correct
        else:
            # Probability of being incorrect given mastery
            p_incorrect = p_mastery * self.p_slip + (1 - p_mastery) * (1 - self.p_guess)
            p_mastery_given_obs = (p_mastery * self.p_slip) / p_incorrect

        # Update mastery with learning transition
        new_mastery = p_mastery_given_obs + (1 - p_mastery_given_obs) * self.p_transit
        self.mastery[skill] = min(1.0, new_mastery)  # Ensure mastery never exceeds 1.0

    def get_mastery(self, skill):
        """Returns the current mastery probability of a given skill."""
        return self.mastery.get(skill, None)

class AdaptiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AdaptiveLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        if len(out.shape) == 2:
            out = out.unsqueeze(1)
        out = self.fc(out[:, -1, :]) if out.dim() == 3 else self.fc(out)
        return out  

# Load the trained dyscalculia model and scaler
dyscalculia_model = joblib.load("dyscalculia_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_dyscalculia(student_features):
    """Predicts dyscalculia probability based on student features."""
    student_features = scaler.transform([student_features])  # Normalize input
    return dyscalculia_model.predict_proba(student_features)[0][1]  # Return probability

def predict_next_step(model, student_sequence):
    """Predicts the next difficulty and skill using the LSTM model."""
    model.eval()
    with torch.no_grad():
        sequence_tensor = torch.tensor(student_sequence, dtype=torch.float32).unsqueeze(0)
        logits = model(sequence_tensor).squeeze()
        next_difficulty = min(2, max(0, int(round(logits[0].item()))))
        next_skill = min(3, max(0, int(round(logits[1].item()))))
    return next_difficulty, next_skill

def generate_question(skill, difficulty):
    ranges = {"easy": (1, 10), "medium": (10, 100), "hard": (100, 1000)}
    min_val, max_val = ranges[difficulty]
    num1 = random.randint(min_val, max_val)
    num2 = random.randint(min_val, max_val)

    if skill == "addition":
        return f"{num1} + {num2}", num1 + num2
    elif skill == "subtraction":
        return f"{num1} - {num2}", num1 - num2
    elif skill == "multiplication":
        return f"{num1} × {num2}", num1 * num2
    elif skill == "division":
        while num1 % num2 != 0:
            num1 = random.randint(min_val, max_val)
            num2 = random.randint(min_val, max_val)
        return f"{num1} ÷ {num2}", num1 // num2

def provide_step_by_step_solution(question):
    steps = []
    if "+" in question:
        operation = "+"
    elif "-" in question:
        operation = "-"
    elif "×" in question:
        operation = "×"
    elif "÷" in question:
        operation = "÷"
    else:
        steps.append("Invalid question format.")
        return steps

    numbers = list(map(int, question.split(operation)))

    if operation == "+":
        steps.extend(addition_steps(numbers))
    elif operation == "-":
        steps.extend(subtraction_steps(numbers))
    elif operation == "×":
        steps.extend(multiplication_steps(numbers))
    elif operation == "÷":
        steps.extend(division_steps(numbers))

    return steps

def addition_steps(numbers):
    steps = []
    num1, num2 = numbers
    tens1, units1 = divmod(num1, 10)
    tens2, units2 = divmod(num2, 10)

    steps.append("Step 1: Break down the numbers into place values.")
    steps.append(f"Number 1: Tens = {tens1}, Units = {units1}")
    steps.append(f"Number 2: Tens = {tens2}, Units = {units2}")

    units_sum = units1 + units2
    carry = units_sum // 10
    steps.append(f"Step 2: Add the units place: {units1} + {units2} = {units_sum}.")
    if carry > 0:
        steps.append(f"Step 3: Carry over {carry} to the tens place.")

    tens_sum = tens1 + tens2 + carry
    steps.append(f"Step 4: Add the tens place: {tens1} + {tens2} = {tens1 + tens2}.")
    if carry > 0:
        steps.append(f"Step 5: Add the carried-over {carry} to the tens: {tens1 + tens2} + {carry} = {tens_sum}.")

    final_answer = tens_sum * 10 + (units_sum % 10)
    steps.append(f"Step 6: Combine the results: Tens = {tens_sum * 10}, Units = {units_sum % 10}.")
    steps.append(f"Final Answer: {final_answer}")
    return steps

# Similar adjustments for subtraction_steps, multiplication_steps, division_steps...

def subtraction_steps(numbers):
    steps = []
    steps.append("Incorrect.")
    steps.append("\nStep-by-Step Solution (Subtraction):")

    num1, num2 = numbers
    # Step 1: Break down numbers into place values
    tens1, units1 = divmod(num1, 10)
    tens2, units2 = divmod(num2, 10)
    steps.append(f"Step 1: Break down the numbers into place values.")
    steps.append(f"Number 1: Tens = {tens1}, Units = {units1}")
    steps.append(f"Number 2: Tens = {tens2}, Units = {units2}")

    # Step 2: Subtract units place
    if units1 < units2:
        steps.append(f"Step 2: Borrow 1 from the tens place.")
        units1 += 10
        tens1 -= 1
    units_diff = units1 - units2
    steps.append(f"Step 3: Subtract the units place: {units1} - {units2} = {units_diff}.")

    # Step 3: Subtract tens place
    tens_diff = tens1 - tens2
    steps.append(f"Step 4: Subtract the tens place: {tens1} - {tens2} = {tens_diff}.")

    # Step 4: Combine results
    final_answer = tens_diff * 10 + units_diff
    steps.append(f"Step 5: Combine the results: Tens = {tens_diff * 10}, Units = {units_diff}.")
    steps.append(f"\nFinal Answer: {final_answer}")

    return steps

def multiplication_steps(numbers):
    steps = []
    steps.append("Incorrect.")
    steps.append("\nStep-by-Step Solution (Multiplication):")

    num1, num2 = numbers
    # Step 1: Break down numbers into place values
    tens1, units1 = divmod(num1, 10)
    tens2, units2 = divmod(num2, 10)
    steps.append(f"Step 1: Break down the numbers into place values.")
    steps.append(f"Number 1: Tens = {tens1}, Units = {units1}")
    steps.append(f"Number 2: Tens = {tens2}, Units = {units2}")

    # Step 2: Calculate cross-products
    steps.append("\nStep 2: Calculate individual products:")
    units_product = units1 * units2
    steps.append(f"Units place: {units1} * {units2} = {units_product}")

    cross_product1 = tens1 * units2
    steps.append(f"Tens of Number 1 * Units of Number 2: {tens1} * {units2} = {cross_product1}")

    cross_product2 = units1 * tens2
    steps.append(f"Units of Number 1 * Tens of Number 2: {units1} * {tens2} = {cross_product2}")

    tens_product = tens1 * tens2
    steps.append(f"Tens place: {tens1} * {tens2} = {tens_product}")

    # Step 3: Add up the products
    total = units_product + (cross_product1 * 10) + (cross_product2 * 10) + (tens_product * 100)
    steps.append("\nStep 3: Add all products to combine:")
    steps.append(f"Final Sum: {units_product} + {cross_product1 * 10} + {cross_product2 * 10} + {tens_product * 100} = {total}")

    # Final Answer
    steps.append(f"\nFinal Answer: {total}")

    return steps

def division_steps(numbers):
    steps = []
    num1, num2 = numbers
    exact_result = round(num1 / num2, 2)

    steps.append("Incorrect.")
    steps.append("\nStep-by-Step Solution (Division):")
    
    # Step 1: Find quotient and remainder
    quotient = num1 // num2
    remainder = num1 % num2
    steps.append(f"Step 1: Divide {num1} by {num2}.")
    steps.append(f"Quotient = {quotient}, Remainder = {remainder}")

    if remainder == 0:
        steps.append(f"\nFinal Answer: {exact_result}")
        return steps
    
    # Step 2: Represent the equation
    steps.append(f"Step 2: Represent as {num1} = ({quotient} * {num2}) + {remainder}.")
    
    # Step 3: Write remainder as fraction
    steps.append(f"Step 3: Write the remainder as a fraction: {remainder}/{num2}.")
    
    # Step 4: Convert fraction to decimal
    decimal_part = remainder / num2
    steps.append(f"Step 4: Convert the fraction to decimal: {remainder} ÷ {num2} = {decimal_part:.2f}.")
    
    # Step 5: Add quotient and decimal
    steps.append(f"Step 5: Add the quotient and decimal: {quotient} + {decimal_part:.2f} = {exact_result}.")
    
    steps.append(f"\nFinal Answer: {exact_result}")
    return steps

# =========================
# CASES Def
# =========================

def extract_operands(question):
    """
    Extracts operands from a mathematical question string.

    Args:
        question (str): A string containing a mathematical question, e.g., "5 + 3" or "7 * 4".

    Returns:
        tuple: A tuple containing the two operands as integers (num1, num2).
    """
    # Split the question into parts using spaces
    parts = question.split()
    # Extract the first and third elements (operands)
    num1 = int(parts[0])  # Convert the first number to an integer
    num2 = int(parts[2])  # Convert the second number to an integer
    return num1, num2

# Case 1: Treating Addition as Multiplication (Semantic Memory)
def is_addition_as_multiplication(question, student_answer):
    num1, num2 = extract_operands(question)
    multiplied = num1 * num2
    return student_answer == multiplied

# Case 2: Visio-Spatial Memory (Carry Omission)
    
    # Carry Omission: Ignore carry in addition
def is_carry_omission_addition(question, student_answer):
    num1, num2 = extract_operands(question)
    sum_without_carry = 0
    place_value = 1
    while num1 > 0 or num2 > 0:
        digit_sum = (num1 % 10) + (num2 % 10)
        sum_without_carry += (digit_sum % 10) * place_value
        num1 //= 10
        num2 //= 10
        place_value *= 10
    return student_answer == sum_without_carry


    # Concatenation of Partial Sums in Addition
def is_partial_sum_concatenation_addition(question, student_answer):
    num1, num2 = extract_operands(question)
    concatenated_sums = ''
    while num1 > 0 or num2 > 0:
        digit_sum = (num1 % 10) + (num2 % 10)
        concatenated_sums = str(digit_sum) + concatenated_sums
        num1 //= 10
        num2 //= 10
    return student_answer == int(concatenated_sums)

# Borrow Cases for Subtraction
# Borrow Omission (Ignore borrowing in subtraction)

def is_borrow_omission_subtraction(question, student_answer):
    num1, num2 = extract_operands(question)
    diff_without_borrow = 0
    place_value = 1
    while num1 > 0 or num2 > 0:
        digit_diff = (num1 % 10) - (num2 % 10)
        if digit_diff < 0:
            digit_diff += 10
        diff_without_borrow += digit_diff * place_value
        num1 //= 10
        num2 //= 10
        place_value *= 10
    return student_answer == diff_without_borrow

    # Concatenation of Differences in Subtraction
def is_partial_difference_concatenation_subtraction(question, student_answer):
    num1, num2 = extract_operands(question)
    concatenated_differences = ''
    while num1 > 0 or num2 > 0:
        digit_diff = (num1 % 10) - (num2 % 10)
        concatenated_differences = str(abs(digit_diff)) + concatenated_differences
        num1 //= 10
        num2 //= 10
    return student_answer == int(concatenated_differences)

# Case 3: Disorganized Arithmetic Strategy (Procedural Memory)
def is_disorganized_arithmetic(question, student_answer):
    num1, num2 = extract_operands(question)
    disorganized_sum = sum(int(d1) + int(d2) for d1, d2 in zip(str(num1), str(num2)))
    return student_answer == disorganized_sum

# Case 4: Vanishing Digits in Addition/Subtraction
def is_vanishing_digits(question, student_answer):
    correct_answer = eval(question)
    correct_digits = set(str(correct_answer))
    student_digits = set(str(student_answer))
    return student_digits.issubset(correct_digits) and len(student_digits) < len(correct_digits)

# Case 5: Zero Misconception (Semantic Memory)
def is_zero_misconception(question, student_answer):
    # Extract operands from the question
    num1, num2 = extract_operands(question)

    # Calculate the correct answer based on the operator
    if '+' in question:
        correct_answer = num1 + num2
    elif '-' in question:
        correct_answer = num1 - num2
    elif '×' in question:
        correct_answer = num1 * num2
    elif '÷' in question:
        if num2 != 0:  # Prevent division by zero
            correct_answer = num1 // num2
        else:
            return False  # Division by zero invalid question
    else:
        return False  # Unsupported operation

    # Check if any operand contains a 0
    if '0' in str(num1) or '0' in str(num2):
        # Check if the student's answer contains 0 but the correct answer does not
        if '0' in str(student_answer) and '0' not in str(correct_answer):
            return True

    return False

