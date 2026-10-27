// frontend/src/App.tsx
import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { Navbar } from './components/Navbar';

import { Overview } from './pages/Overview';
import { LiveEEG } from './pages/LiveEEG';
import { DatasetExplorer } from './pages/DatasetExplorer';
import { EEGPreprocessing } from './pages/EEGPreprocessing';
import { ModelManagement } from './pages/ModelManagement';
import { ModelInference } from './pages/ModelInference';
import { TrialMinimisation } from './pages/TrialMinimisation';
import { GANAugmentation } from './pages/GANAugmentation';
import { ModelComparison } from './pages/ModelComparison';
import { ResultsReports } from './pages/ResultsReports';
import { HardwareConnections } from './pages/HardwareConnections';
import { Settings } from './pages/Settings';
import { Diagnostics } from './pages/Diagnostics';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-slate-950 text-slate-100">
        {/* Left Sidebar */}
        <Sidebar />

        {/* Main Content Area */}
        <div className="flex-1 ml-64 flex flex-col min-h-screen">
          <Navbar />
          <main className="flex-1">
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/live" element={<LiveEEG />} />
              <Route path="/datasets" element={<DatasetExplorer />} />
              <Route path="/preprocessing" element={<EEGPreprocessing />} />
              <Route path="/models" element={<ModelManagement />} />
              <Route path="/inference" element={<ModelInference />} />
              <Route path="/minimisation" element={<TrialMinimisation />} />
              <Route path="/gan" element={<GANAugmentation />} />
              <Route path="/comparison" element={<ModelComparison />} />
              <Route path="/results" element={<ResultsReports />} />
              <Route path="/hardware" element={<HardwareConnections />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/diagnostics" element={<Diagnostics />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
};

export default App;
