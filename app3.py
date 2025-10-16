from flask import Flask, request, Response, jsonify
from temp2 import generate_question, provide_step_by_step_solution, predict_next_step
from flask_cors import CORS
import torch
from temp2 import AdaptiveLSTM, SimpleBKT  # Replace with your actual model class
import random
from temp2 import (
    is_addition_as_multiplication, is_carry_omission_addition, 
    is_partial_sum_concatenation_addition, is_borrow_omission_subtraction, 
    is_partial_difference_concatenation_subtraction, is_disorganized_arithmetic, 
    is_vanishing_digits, is_zero_misconception
)
from temp2 import predict_dyscalculia
import numpy as np
import copy


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})
# CORS(app, resources={r"/*": {"origins": "*"}})
# @app.after_request
# def add_cors_headers(response):
#     response.headers["Access-Control-Allow-Origin"] = "*"
#     response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
#     response.headers["Access-Control-Allow-Headers"] = "Content-Type"
#     return response

# Load the pre-trained LSTM model
model = AdaptiveLSTM(5, 50, 2)  # Initialize the model class
model.load_state_dict(torch.load("lstm_model2.pth"))  # Load the saved weights
model.eval()  # Set model to evaluation mode

skills = ["addition", "subtraction", "multiplication", "division"]
# BKT
bkt_model = SimpleBKT(skills)


# Quiz state to track progress
quiz_state = {
    "current_index": 0,
    "total_questions": 20,
    "correct_answers": 0,
    "skills": ["addition", "subtraction", "multiplication", "division"],
    "questions": [],
    "student_data": [],  # Store response times, correctness, etc., for LSTM input
    "response_times": [],
    "skill_errors": {skill: 0 for skill in ["addition", "subtraction", "multiplication", "division"]},
    "case_counts": {
        "case1": 0,
        "case2": 0,
        "case3": 0,
        "case4": 0,
        "case5": 0,
    }
}

# quiz_state_reset = quiz_state
quiz_state_reset = copy.deepcopy(quiz_state)


@app.route('/api/get-question', methods=['GET'])
def get_question():
    if quiz_state["current_index"] >= quiz_state["total_questions"]:
        return jsonify({"quiz_complete": True})  # Indicate the quiz is finished
    
    if "skill" not in quiz_state:
        quiz_state["skill"] = 0
        
    if "attempt_in_skill" not in quiz_state:
        quiz_state["attempt_in_skill"] = 0

    # Predict skill and difficulty using LSTM if student data exists
    student_sequence = quiz_state["student_data"]

    # attempt_in_skill, skill = 0, 0
    skill = quiz_state["skill"]
    attempt_in_skill = quiz_state["attempt_in_skill"]

    # print("skill",skill)
    if not student_sequence:
        difficulty = 0  # Default to easy addition for the first question
        # print("hereeee")
    else:
        # difficulty, skill = predict_next_step(model, student_sequence)
        if attempt_in_skill < 5:
            difficulty, _ = predict_next_step(model, student_sequence)
            # print(student_sequence)
        else:
            # skill, difficulty, attempt_in_skill = (skill + 1) % 4, 0, 0
            skill, difficulty, attempt_in_skill = (skill + 1) % len(quiz_state["skills"]), 0, 0
            quiz_state["skill"] = skill
            # quiz_state["attempt_in_skill"] = attempt_in_skill

    # print(attempt_in_skill)
    # Map skill and difficulty levels

    # skills = ["addition", "subtraction", "multiplication", "division"]
    difficulty_levels = ["easy", "medium", "hard"]

    # skill_str = skills[skill]
    skill_str = quiz_state["skills"][skill]
    difficulty_str = difficulty_levels[difficulty]

    # Generate question
    question, correct_answer = generate_question(skill_str, difficulty_str)
    print(question)
    quiz_state["questions"].append({"question": question, "correctAnswer": correct_answer, "difficulty":difficulty, "skill": skill})
    attempt_in_skill += 1
    quiz_state["attempt_in_skill"] = attempt_in_skill
    
    print(question)
    return jsonify({"question": question, "quiz_complete": False, "question_no": quiz_state["current_index"]+1})


@app.route('/api/submit-answer', methods=['POST'])
def submit_answer():
    data = request.json
    question = data["question"]
    user_answer = data["userAnswer"]
    response_time = data.get("responseTime", 0)
    quiz_state["response_times"].append(response_time)

    question_data = quiz_state["questions"][quiz_state["current_index"]]
    is_correct = int(user_answer) == int(question_data["correctAnswer"])

    # skills = ["addition", "subtraction", "multiplication", "division"]
    skillname = quiz_state["skills"][quiz_state["skill"]]
    quiz_state["skill_errors"][skillname] += 0 if is_correct else 1

    # Update BKT model with the response
    bkt_model.update(skillname, is_correct)
    mastery = bkt_model.get_mastery(skillname)  # Get updated mastery


    # Update quiz state
    quiz_state["current_index"] += 1
    quiz_state["correct_answers"] += int(is_correct)
    quiz_state["student_data"].append([response_time, int(is_correct), 1 if is_correct else random.randint(2, 4), 
                                       question_data["difficulty"], question_data["skill"]])
    if not is_correct:
        if is_addition_as_multiplication(question, user_answer):
            quiz_state["case_counts"]["case1"] += 1
        elif is_carry_omission_addition(question, user_answer):
            quiz_state["case_counts"]["case2"] += 1
        elif is_partial_sum_concatenation_addition(question, user_answer):
            quiz_state["case_counts"]["case2"] += 1
        elif is_borrow_omission_subtraction(question, user_answer):
            quiz_state["case_counts"]["case2"] += 1
        elif is_partial_difference_concatenation_subtraction(question, user_answer):
            quiz_state["case_counts"]["case2"] += 1
        elif is_disorganized_arithmetic(question, user_answer):
            quiz_state["case_counts"]["case3"] += 1
        elif is_vanishing_digits(question.replace('×', '*') if "×" in question else question.replace('÷', '/'), user_answer):
            quiz_state["case_counts"]["case4"] += 1
        elif is_zero_misconception(question, user_answer):
            quiz_state["case_counts"]["case5"] += 1

    # Return quiz completion status
    quiz_complete = quiz_state["current_index"] >= quiz_state["total_questions"]
    # print(quiz_complete)

    if quiz_complete:

        avg_response_time = np.mean(quiz_state["response_times"])
        weakest_op = max(quiz_state["skill_errors"], key=quiz_state["skill_errors"].get)

        student_features = [
            20,
            avg_response_time,
            quiz_state["case_counts"]["case1"],
            quiz_state["case_counts"]["case2"],
            quiz_state["case_counts"]["case3"],
            quiz_state["case_counts"]["case4"],
            quiz_state["case_counts"]["case5"]
        ]
        dys_prob = predict_dyscalculia(student_features)

        quiz_state["masteries"] = {skill: bkt_model.get_mastery(skill) for skill in skills}
        has_low_mastery = any(mastery < 0.8 for mastery in quiz_state["masteries"].values())

        return jsonify(
            {
                "result": "Correct" if is_correct else "Incorrect", 
                "total_correct": quiz_state["correct_answers"], 
                "avg_resp_time": avg_response_time, 
                "dys_prob": dys_prob, 
                "weakest_operation": weakest_op,
                "cases_data": quiz_state["case_counts"],
                "updated_mastery": quiz_state["masteries"],
                "has_low_mastery": has_low_mastery,
                "quiz_complete": quiz_complete
            }
        )

    return jsonify({"result": "Correct" if is_correct else "Incorrect", "quiz_complete": quiz_complete})


@app.route('/api/get-solution', methods=['POST'])
def get_solution():
    data = request.json
    question = data["question"]
    steps = provide_step_by_step_solution(question)
    return jsonify({"steps": steps})

@app.route('/api/start-new-quiz', methods=['GET'])
def start_new_quiz():
    global quiz_state
    # masteries = {skill: bkt_model.get_mastery(skill) for skill in skills}
    masteries = quiz_state["masteries"]
    weak_skills = [skill for skill, mastery in masteries.items() if mastery < 0.8]
    # quiz_state = quiz_state_reset
    quiz_state = copy.deepcopy(quiz_state_reset)
    quiz_state["skills"] = weak_skills
    quiz_state["total_questions"] = len(weak_skills)*5
    return Response(status=204)

if __name__ == '__main__':
    app.run(debug=True)
