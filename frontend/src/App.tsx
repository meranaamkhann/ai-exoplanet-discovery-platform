import { BrowserRouter, Routes, Route } from 'react-router-dom';
import AppLayout from './components/AppLayout';
import DashboardPage from './pages/DashboardPage';
import AnalyzePage from './pages/AnalyzePage';
import ResultsPage from './pages/ResultsPage';
import HistoryPage from './pages/HistoryPage';
import CandidatesPage from './pages/CandidatesPage';
import RealDataPage from './pages/RealDataPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/analyze" element={<AnalyzePage />} />
          <Route path="/results/:analysisId" element={<ResultsPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/candidates" element={<CandidatesPage />} />
          <Route path="/real-data" element={<RealDataPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
