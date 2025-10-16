# NEW SHIT (AVERAGE RES Time and Prediction) + CASES
import joblib
import random
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ==============================
# 1️⃣ GENERATE SYNTHETIC DATA
# ==============================

def generate_synthetic_data(num_students=500, questions_per_student=20):
    """Generates structured quiz data for training the LSTM model and dyscalculia prediction."""
    data, labels, dyscalculia_features, dyscalculia_labels = [], [], [], []

    for student_id in range(1, num_students + 1):
        skill = 0  # Start with addition
        difficulty = 0  # Start with easy level
        attempts_in_skill = 0  # Track per skill

        total_wrong = 0
        total_correct = 0
        avg_response_time = 0
        response_times = []
        skill_errors = [0, 0, 0, 0]  # Track errors per operation (Addition, Subtraction, Multiplication, Division)

        for _ in range(questions_per_student):
            correct = np.random.choice([1, 0], p=[0.7, 0.3] if difficulty == 0 else [0.5, 0.5] if difficulty == 1 else [0.3, 0.7])
            response_time = round(random.uniform(1, 15), 2)
            attempt_count = 1 if correct else random.randint(2, 4)

            data.append([response_time, correct, attempt_count, difficulty, skill])
            labels.append([min(2, max(0, difficulty + 1 if correct else difficulty - 1)), skill])

            response_times.append(response_time)
            if correct:
                total_correct += 1
            else:
                total_wrong += 1
                skill_errors[skill] += 1

            attempts_in_skill += 1
            if attempts_in_skill >= 5:
                skill = (skill + 1) % 4
                difficulty = 0
                attempts_in_skill = 0

        avg_response_time = sum(response_times) / len(response_times)

        # Store per-student dyscalculia features
        dyscalculia_features.append([
            total_correct,  # Total correct answers
            np.mean(response_times),  # Average response time
            max(response_times),  # Maximum response time
            sum(skill_errors),  # Total mistakes across all operations
            max(skill_errors)  # Maximum mistakes in a single operation
        ])

        # Assign dyscalculia label: (More than 10 mistakes OR avg response time > 6 sec)
        dyscalculia_labels.append(1 if total_wrong > 10 or avg_response_time > 6 else 0)
    
    # pd.DataFrame(data, columns=["StudentID", "ResponseTime", "Correct", "AttemptCount", "Difficulty", "Skill"]).to_csv("synthetic_data.csv", index=False)
    return np.array(data), np.array(labels), np.array(dyscalculia_features), np.array(dyscalculia_labels)

# ==============================
# 2️⃣ LSTM MODEL
# ==============================

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

# ==============================
# 3️⃣ TRAINING THE LSTM
# ==============================

class StudentDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sequence, label

def train_lstm_model(model, train_loader, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            predictions = model(sequences)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "lstm_model2.pth")

# ==============================
# 4️⃣ TRAIN DYSLEXIA PREDICTION MODEL
# ==============================

def train_dyscalculia_model(dyscalculia_features, dyscalculia_labels):
    scaler = StandardScaler()
    features = scaler.fit_transform(dyscalculia_features)  # Normalize
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, dyscalculia_labels)
    joblib.dump(model, "dyscalculia_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    return model, scaler


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


# ==============================
# 5️⃣ QUIZ SYSTEM WITH REPORT
# ==============================

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
    # Determine the operation based on symbols in the string
    if "+" in question:
        operation = "+"
    elif "-" in question:
        operation = "-"
    elif "×" in question:
        operation = "×"
    elif "÷" in question:
        operation = "÷"
    else:
        print("Invalid question format.")
        return

    # Split the question into numbers
    numbers = list(map(int, question.split(operation)))
    
    # Calculate the correct answer
    if operation == "+":
        correct_answer = numbers[0] + numbers[1]
        addition_steps(numbers)
    elif operation == "-":
        correct_answer = numbers[0] - numbers[1]
        subtraction_steps(numbers)
    elif operation == "×":
        correct_answer = numbers[0] * numbers[1]
        multiplication_steps(numbers)
    elif operation == "÷":
        correct_answer = numbers[0] // numbers[1]  # Assuming integer division
        division_steps(numbers)

    # Print the final answer
    # print(f"\nFinal Answer: {correct_answer}")

def addition_steps(numbers):
    num1, num2 = numbers
    # print(f"Question: {num1} + {num2}")
    # user_answer = int(input("Your Answer: "))
    # if user_answer == num1 + num2:
    #     print("Correct!")
    # else:
    print("Incorrect.")
    print("\nStep-by-Step Solution (Addition):")
    # Step 1: Break down numbers into place values
    tens1, units1 = divmod(num1, 10)
    tens2, units2 = divmod(num2, 10)
    print(f"Step 1: Break down the numbers into place values.")
    print(f"Number 1: Tens = {tens1}, Units = {units1}")
    print(f"Number 2: Tens = {tens2}, Units = {units2}")

    # Step 2: Add units place
    units_sum = units1 + units2
    carry = units_sum // 10  # Carry from the units place
    print(f"Step 2: Add the units place: {units1} + {units2} = {units_sum}.")
    if carry > 0:
        print(f"Step 3: Carry over {carry} to the tens place.")

    # Step 3: Add tens place
    tens_sum = tens1 + tens2 + carry
    print(f"Step 4: Add the tens place: {tens1} + {tens2} = {tens1 + tens2}.")
    if carry > 0:
        print(f"Step 5: Add the carried-over {carry} to the tens: {tens1 + tens2} + {carry} = {tens_sum}.")

    # Step 4: Combine results
    final_answer = tens_sum * 10 + (units_sum % 10)
    print(f"Step 6: Combine the results: Tens = {tens_sum * 10}, Units = {units_sum % 10}.")
    print(f"\nFinal Answer: {final_answer}")

def subtraction_steps(numbers):
    num1, num2 = numbers
    # print(f"Question: {num1} - {num2}")
    # user_answer = int(input("Your Answer: "))
    # if user_answer == num1 - num2:
    #     print("Correct!")
    # else:
    print("Incorrect.")
    print("\nStep-by-Step Solution (Subtraction):")
    # Step 1: Break down numbers into place values
    tens1, units1 = divmod(num1, 10)
    tens2, units2 = divmod(num2, 10)
    print(f"Step 1: Break down the numbers into place values.")
    print(f"Number 1: Tens = {tens1}, Units = {units1}")
    print(f"Number 2: Tens = {tens2}, Units = {units2}")

    # Step 2: Subtract units place
    if units1 < units2:
        print(f"Step 2: Borrow 1 from the tens place.")
        units1 += 10
        tens1 -= 1
    units_diff = units1 - units2
    print(f"Step 3: Subtract the units place: {units1} - {units2} = {units_diff}.")

    # Step 3: Subtract tens place
    tens_diff = tens1 - tens2
    print(f"Step 4: Subtract the tens place: {tens1} - {tens2} = {tens_diff}.")

    # Step 4: Combine results
    final_answer = tens_diff * 10 + units_diff
    print(f"Step 5: Combine the results: Tens = {tens_diff * 10}, Units = {units_diff}.")
    print(f"\nFinal Answer: {final_answer}")

def multiplication_steps(numbers):
    num1, num2 = numbers
    # print(f"Question: {num1} * {num2}")
    # user_answer = int(input("Your Answer: "))
    # if user_answer == num1 * num2:
    #     print("Correct!")
    # else:
    print("Incorrect.")
    print("\nStep-by-Step Solution (Multiplication):")
    
    # Step 1: Break down numbers into place values
    tens1, units1 = divmod(num1, 10)
    tens2, units2 = divmod(num2, 10)
    print(f"Step 1: Break down the numbers into place values.")
    print(f"Number 1: Tens = {tens1}, Units = {units1}")
    print(f"Number 2: Tens = {tens2}, Units = {units2}")

    # Step 2: Calculate cross-products
    print("\nStep 2: Calculate individual products:")
    units_product = units1 * units2
    print(f"Units place: {units1} * {units2} = {units_product}")

    cross_product1 = tens1 * units2
    print(f"Tens of Number 1 * Units of Number 2: {tens1} * {units2} = {cross_product1}")

    cross_product2 = units1 * tens2
    print(f"Units of Number 1 * Tens of Number 2: {units1} * {tens2} = {cross_product2}")

    tens_product = tens1 * tens2
    print(f"Tens place: {tens1} * {tens2} = {tens_product}")

    # Step 3: Add up the products
    total = units_product + (cross_product1 * 10) + (cross_product2 * 10) + (tens_product * 100)
    print("\nStep 3: Add all products to combine:")
    print(f"Final Sum: {units_product} + {cross_product1 * 10} + {cross_product2 * 10} + {tens_product * 100} = {total}")

    # Final Answer
    print(f"\nFinal Answer: {total}")

def division_steps(numbers):
    num1, num2 = numbers
    # print(f"Question: {num1} / {num2}")
    # user_answer = float(input("Your Answer: "))
    exact_result = round(num1 / num2, 2)

    # if round(user_answer, 2) == exact_result:
    #     print("Correct!")
    # else:
    print("Incorrect.")
    print("\nStep-by-Step Solution (Division):")
    # Step 1: Find quotient and remainder
    quotient = num1 // num2
    remainder = num1 % num2
    print(f"Step 1: Divide {num1} by {num2}.")
    print(f"Quotient = {quotient}, Remainder = {remainder}")

    if remainder == 0:
        print(f"\nFinal Answer: {exact_result}")
        return
    
    # Step 2: Represent the equation
    print(f"Step 2: Represent as {num1} = ({quotient} * {num2}) + {remainder}.")
    
    # Step 3: Write remainder as fraction
    print(f"Step 3: Write the remainder as a fraction: {remainder}/{num2}.")
    
    # Step 4: Convert fraction to decimal
    decimal_part = remainder / num2
    print(f"Step 4: Convert the fraction to decimal: {remainder} ÷ {num2} = {decimal_part:.2f}.")
    
    # Step 5: Add quotient and decimal
    print(f"Step 5: Add the quotient and decimal: {quotient} + {decimal_part:.2f} = {exact_result}.")
    
    print(f"\nFinal Answer: {exact_result}")



def run_adaptive_quiz(model, dyscalculia_model, scaler):
    case1_count = case2_add1_count = case2_add2_count = case2_sub1_count = case2_sub2_count = case3_count = case4_count = case5_count = 0

    skills = ["addition", "subtraction", "multiplication", "division"]
    difficulty_levels = ["easy", "medium", "hard"]

    student_data = []
    current_skill, current_difficulty, attempts_in_skill = 0, 0, 0
    total_correct, response_times, skill_errors = 0, [], {skill: 0 for skill in skills}

    for i in range(20):
        skill = skills[current_skill]
        difficulty = difficulty_levels[current_difficulty]
        question, answer = generate_question(skill, difficulty)

        print(f"\nQuestion {i + 1}: {question}")
        start_time = time.time()
        user_answer = int(input("Your answer: "))
        response_time = time.time() - start_time

        correct = user_answer == answer
        response_times.append(response_time)
        skill_errors[skill] += 0 if correct else 1
        total_correct += correct

        # ========================
        # SOLUTION INTEGRATION
        # ========================
        if correct:
            print("Correct Answer!")
        else:
            provide_step_by_step_solution(question)

        # student_data.append([response_time, int(correct), 1 if correct else random.randint(2, 4), current_difficulty, current_skill])
        student_data.append([response_time, int(correct), 1 if correct else random.randint(2, 4), current_difficulty, current_skill])
        attempts_in_skill += 1

        # cases
        # addition
        if skill == "addition":
            if is_addition_as_multiplication(question, user_answer):
                case1_count += 1

            elif is_carry_omission_addition(question, user_answer):
                case2_add1_count += 1
            
            elif is_partial_sum_concatenation_addition(question, user_answer):
                case2_add2_count += 1
            
            elif is_disorganized_arithmetic(question, user_answer):
                case3_count += 1
            
            elif is_vanishing_digits(question, user_answer):
                case4_count += 1

        # subtraction
        elif skill == "subtraction":
            if is_borrow_omission_subtraction(question, user_answer):
                case2_sub1_count += 1
            
            elif is_partial_difference_concatenation_subtraction(question, user_answer):
                case2_sub2_count += 1

            elif is_vanishing_digits(question, user_answer):
                case4_count += 1
        
        if is_zero_misconception(question, user_answer):
                case5_count += 1
        

        if attempts_in_skill < 5:
            current_difficulty, _ = predict_next_step(model, student_data)
        else:
            print("\n⏩ Moving to the next skill.")
            current_skill, current_difficulty, attempts_in_skill = (current_skill + 1) % 4, 0, 0
    
    # student_features = scaler.transform([[
    #     total_correct, np.mean(response_times), max(response_times), sum(skill_errors.values()), max(skill_errors.values())
    # ]])
    # dyscalculia_prob = dyscalculia_model.predict_proba(student_features)[0][1]


    # Prepare features for dyscalculia model
    addition_as_multiplication = case1_count
    carry_omission = case2_add1_count + case2_add2_count + case2_sub1_count + case2_sub2_count
    disorganized_strategy = case3_count
    vanishing_digits = case4_count
    zero_misconception = case5_count

    student_features = scaler.transform([[
        20,
        np.mean(response_times),  
        addition_as_multiplication,  
        carry_omission,  
        disorganized_strategy,  
        vanishing_digits,  
        zero_misconception
    ]])
    dyscalculia_prob = dyscalculia_model.predict_proba(student_features)[0][1]

    # student_features = scaler.transform([[
    #     np.mean(response_times), 
    #     addition_as_multiplication, 
    #     carry_omission, 
    #     disorganized_strategy, 
    #     vanishing_digits, 
    #     zero_misconception
    # ]])

    # dyscalculia_prob = dyscalculia_model.predict_proba(student_features)[0][1]
    # label_map = {0: "Not Dyscalculic", 1: "Dyscalculic"}
    # dyscalculia_label = dyscalculia_model.predict(student_features)[0]
    # label_description = label_map[dyscalculia_label]



    print("\n🏆 Quiz Complete! Here's your performance report:")
    print(f"- Accuracy: {total_correct}/20 ({(total_correct/20) * 100:.2f}%)")
    print(f"- Average Response Time: {np.mean(response_times):.2f} sec")
    print(f"- Probability of Dyscalculia: {dyscalculia_prob:.2%}")
    # print(f"- Probability of Dyscalculia: {label_description}")
    print(f"- Weakest Operation: {max(skill_errors, key=skill_errors.get)}")
    print()
    print(f"- CASE 1: {addition_as_multiplication}")
    print(f"- CASE 2: {carry_omission}")
    print(f"- CASE 3: {disorganized_strategy}")
    print(f"- CASE 4: {vanishing_digits}")
    print(f"- CASE 5: {zero_misconception}")

# ==============================
# 6️⃣ TRAIN & RUN THE QUIZ
# ==============================

# dys_data = pd.read_csv('dyscalculia_dataset.csv')
dys_data = pd.read_csv('dyscalculia_labeled.csv')
dys_features = dys_data[['questions_attempted','response_time', 'Addition_as_Multiplication', 'Carry_Omission', 'Disorganized_Strategy', 'Vanishing_Digits', 'Zero_Misconception']]
dys_label = dys_data['dyscalculic']

# Split the data into train and test sets
from sklearn.model_selection import train_test_split
dys_train_features, dys_test_features, dys_train_labels, dys_test_labels = train_test_split(
    dys_features, dys_label, test_size=0.2, random_state=42
)


# scaler = StandardScaler()
# train_features_scaled = scaler.fit_transform(dys_train_features)
# test_features_scaled = scaler.transform(dys_test_features)

data, labels, dyscalculia_features, dyscalculia_labels = generate_synthetic_data()
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)
train_loader = DataLoader(StudentDataset(train_data, train_labels), batch_size=16, shuffle=True)

model = AdaptiveLSTM(5, 50, 2)
train_lstm_model(model, train_loader)

# dyscalculia_model, scaler = train_dyscalculia_model(dyscalculia_features, dyscalculia_labels)
dyscalculia_model, scaler = train_dyscalculia_model(dys_features, dys_label)
run_adaptive_quiz(model, dyscalculia_model, scaler)



'''
# ========== NEW CODE =========
# ==============================
# 7️⃣ INTEGRATE 5 CASES & USE NEW DATASET
# ==============================

# Define new features and labels for dyscalculia based on the updated dataset
def preprocess_new_data(new_data_path):
    """Loads and preprocesses the new dataset with 5 cases."""
    new_data = pd.read_csv(new_data_path)
    # Select features for dyscalculia prediction based on the 5 cases
    features = new_data[[
        "Total Correct Answers",    # Total correct responses
        "Average Response Time",    # Average response time
        "Max Response Time",        # Maximum response time
        "Total Mistakes",           # Total mistakes across all skills
        "Max Mistakes in Skill"     # Maximum mistakes in a single skill
    ]]
    labels = new_data["Dyscalculia Label"]  # Dyscalculia label (1 or 0)
    return features, labels

# Train dyscalculia prediction model using new features
def train_dyscalculia_model_with_new_data(features, labels):
    """Trains the dyscalculia prediction model using the new features."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)  # Normalize new features
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features_scaled, labels)
    return model, scaler

# Load the new dataset and train the updated dyscalculia model
new_features, new_labels = preprocess_new_data("dyscalculia_dataset.csv")  # Replace with your dataset path
new_dyscalculia_model, new_scaler = train_dyscalculia_model_with_new_data(new_features, new_labels)

# Modify the quiz system to use the new dyscalculia model
def run_adaptive_quiz_with_new_model(model, new_dyscalculia_model, new_scaler):
    """Runs the quiz using the updated dyscalculia model with 5 cases."""
    skills = ["addition", "subtraction", "multiplication", "division"]
    difficulty_levels = ["easy", "medium", "hard"]

    student_data = []
    current_skill, current_difficulty, attempts_in_skill = 0, 0, 0
    total_correct, response_times, skill_errors = 0, [], {skill: 0 for skill in skills}

    for i in range(20):
        skill = skills[current_skill]
        difficulty = difficulty_levels[current_difficulty]
        question, answer = generate_question(skill, difficulty)

        print(f"\nQuestion {i + 1}: {question}")
        start_time = time.time()
        user_answer = int(input("Your answer: "))
        response_time = time.time() - start_time

        correct = user_answer == answer
        response_times.append(response_time)
        skill_errors[skill] += 0 if correct else 1
        total_correct += correct

        student_data.append([response_time, int(correct), 1 if correct else random.randint(2, 4), current_difficulty, current_skill])
        attempts_in_skill += 1

        if attempts_in_skill < 5:
            current_difficulty, _ = predict_next_step(model, student_data)
        else:
            print("\n⏩ Moving to the next skill.")
            current_skill, current_difficulty, attempts_in_skill = (current_skill + 1) % 4, 0, 0

    # Use the new dyscalculia model for prediction
    student_features = new_scaler.transform([[
        total_correct, np.mean(response_times), max(response_times), sum(skill_errors.values()), max(skill_errors.values())
    ]])
    dyscalculia_prob = new_dyscalculia_model.predict_proba(student_features)[0][1]

    print("\n🏆 Quiz Complete! Here's your performance report:")
    print(f"- Accuracy: {total_correct}/20 ({(total_correct/20) * 100:.2f}%)")
    print(f"- Average Response Time: {np.mean(response_times):.2f} sec")
    print(f"- Probability of Dyscalculia: {dyscalculia_prob:.2%}")
    print(f"- Weakest Operation: {max(skill_errors, key=skill_errors.get)}")

# Run the quiz using the updated model
run_adaptive_quiz_with_new_model(model, new_dyscalculia_model, new_scaler)
'''