import { useLocation, useNavigate } from "react-router-dom";
import axios from "axios";
import './results.css';

function ResultsPage() {

  const { state } = useLocation();
  const navigate = useNavigate();
  function restartQuiz() {
    axios.get("http://127.0.0.1:5000/api/start-new-quiz")
      .then(() => {
        navigate("/");  // Navigate back to quiz page
      })
      .catch(error => console.error("Error restarting quiz:", error));
  }

  if (!state) return <h2>No results available</h2>;

  return (
    <div className="resMainContainer">
      <div className="resHeader">
        <h2>Quiz Completed!</h2>
        {/* !state.has_low_mastery && <button onClick={restartQuiz}>Exercise Personalized Quiz</button> */}
        {state.has_low_mastery ? (
          <button onClick={restartQuiz}>Exercise Personalized Quiz</button>
        ) : (
          <p>Congratulations! You've mastered all skills. 🎉</p>
        )}
      </div>
      
      <div className="report">
        <div className="resSubHead">Here's your performance report: </div>
        <div className="accuracyData">Accuracy: {state.total_correct}/20 ({(state.total_correct/20)*100}%)</div>
        <div className="avgRespTimeData">Average Response Time: {state.avg_resp_time.toFixed(2)} sec</div>
        <div className="dysProbData">Probability of Dyscalculia: {(state.dys_prob*100).toFixed(2)}%</div>
        <div className="weakestOpData">Weakest Operation: {state.weakest_operation.replace(state.weakest_operation.charAt(0), state.weakest_operation.charAt(0).toUpperCase())}</div>

        <div className="addition">Addition Mastery: {state.updated_mastery["addition"].toFixed(3)}</div>
        <div className="subtraction">Subtraction Mastery: {state.updated_mastery["subtraction"].toFixed(3)}</div>
        <div className="multiplication">Multiplication Mastery: {state.updated_mastery["multiplication"].toFixed(3)}</div>
        <div className="division">Division Mastery: {state.updated_mastery["division"].toFixed(3)}</div>

        <div className="casesData">
            <div className="casesHead">
              <h3>Error Breakdown:</h3>
            </div>
            <table border="1" className="casesTable">
                <thead>
                    <tr>
                        <th>Case #</th>
                        <th>Definition</th>
                        <th>Count</th>
                    </tr>
                </thead>
                <tbody>
                    {Object.entries(state.cases_data).map(([key, value]) => (
                        <tr key={key}>
                            <td>{key}</td>
                            <td>{{
                                case1: "Addition mistaken as multiplication",
                                case2: "Carry/borrow omission in multi-digit math",
                                case3: "Disorganized arithmetic strategy",
                                case4: "Forgetting or omitting digits",
                                case5: "Thinking zero in an operation makes result zero"
                            }[key]}</td>
                            <td>{value}</td>
                        </tr>
                    ))}
                </tbody>
            </table>

        </div>
      </div>
    </div>
  );
}

export default ResultsPage;
