import React, { useState, useEffect, useRef } from "react";
import { Routes, Route, useNavigate } from "react-router-dom";
import axios from "axios";
import '../quiz.css'

function QuizApp() {
  const [startTime, setStartTime] = useState(null);
  const [quizComplete, setQuizComplete] = useState(false)
  const [questionNo, setQuestionNo] = useState("")
  const [question, setQuestion] = useState("");
  const [userAnswer, setUserAnswer] = useState("");
  const [result, setResult] = useState("");
  const [solutionSteps, setSolutionSteps] = useState([]);
  const [resultsData, setResultsData] = useState(null);

  const navigate = useNavigate();
  const show_results = () => {
    if (quizComplete) {
      setTimeout(() => {
        navigate("/results");
      }, 5000);
    }
  }

  useEffect(() => {
    if (quizComplete) {
      setTimeout(() => {
        navigate("/results", { state: resultsData });
      }, 5000);
    }
    else{
      fetchQuestion();
    }
  }, [quizComplete]);

  // const fetchQuestion = () => {
  //   axios.get("http://127.0.0.1:5000/api/get-question")
  //     .then(response => setQuestion(response.data.question))
  //     .catch(error => console.error(error));
  // };

  const fetchQuestion = () => {
    axios.get("http://127.0.0.1:5000/api/get-question")
      .then(response => {
        if (response.data.quiz_complete) {
          setQuizComplete(true);
        } else {
          setQuestion(response.data.question);
          setQuestionNo(response.data.question_no);
          setStartTime(performance.now())
          
          document.getElementsByTagName("input")[0].focus();
          setUserAnswer("");
        }
      })
      .catch(error => console.error(error));
  };

  const handleSubmit = () => {
    console.log(userAnswer);
    axios.post("http://127.0.0.1:5000/api/submit-answer", {
      question: question,
      userAnswer: parseInt(userAnswer),
      responseTime: (performance.now() - startTime) / 1000  // Time in seconds
    }).then(response => {
      setResult(response.data.result);
      if (response.data.result === "Incorrect") fetchSolution();
      if(!response.data.quiz_complete) fetchQuestion();
      else{
        setQuizComplete(true);
        setResultsData(response.data);
      } 
    }).catch(error => console.error(error));
  };

  const fetchSolution = () => {
    axios.post("http://127.0.0.1:5000/api/get-solution", {
      question: question,
    }).then(response => setSolutionSteps(response.data.steps))
      .catch(error => console.error(error));
  };

  return (
    <div className="mainContainer">
      <div className="header">
        <h2>EDSense</h2>
      </div>
      <div className="quiz-area">
        <div className="ques-no">
          {question && <h2>Question {questionNo}: </h2>}
        </div>
        <div className="question">
          <div className="num1">{question.split(" ")[0]}</div>
          <div className="op">{question.split(" ")[1]}</div>
          <div className="num2">{question.split(" ")[2]}</div>
          <div className="op">=</div>
          {/* {question}   */}
          <div className="answer">
            <input
              type="number"
              value={userAnswer}
              onChange={(e) => setUserAnswer(e.target.value)}
              // placeholder="Your Answer"
            />
          </div>
        </div>
        <div className="submit-btn">
          <button onClick={handleSubmit} disabled={quizComplete}>Submit</button>
        </div>
        <div className="res">
          {result && <h2 style={{"color": result=="Correct"?"green":"red", display:"inline"}}>{result}</h2>}
        </div>
        <div className="solution">
          {solutionSteps.length > 0 && (
            <div>
              <h3>Step-by-Step Solution:</h3>
              <ul>
                {solutionSteps.map((step, index) => (
                  <li key={index}>{step}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default QuizApp;
