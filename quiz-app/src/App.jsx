import { Routes, Route, useNavigate } from "react-router-dom";
import QuizApp from "./components/QuizApp";
import ResultsPage from "./components/ResultsPage";

function App() {
  return (
    <Routes>
      <Route path="/" element={<QuizApp />} />
      <Route path="/results" element={<ResultsPage />} />
    </Routes>
  );
}

export default App;
