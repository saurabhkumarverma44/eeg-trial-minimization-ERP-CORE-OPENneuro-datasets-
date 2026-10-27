// frontend/src/pages/DatasetExplorer.tsx
import React, { useEffect, useState } from 'react';
import {
  Database,
  Search,
  Filter,
  AlertTriangle,
  CheckCircle2,
  FileCode,
  Layers,
  ArrowRight,
  HardDrive,
  Info,
} from 'lucide-react';
import { fetchDatasets, DatasetItem } from '../api/client';
import { Link } from 'react-router-dom';

export const DatasetExplorer: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedDataset, setSelectedDataset] = useState<DatasetItem | null>(null);

  useEffect(() => {
    fetchDatasets()
      .then((data) => {
        setDatasets(data);
        if (data.length > 0) {
          const ds4 = data.find((d) => d.dataset_id.includes('gonogo_4')) || data[0];
          setSelectedDataset(ds4);
        }
      })
      .catch((err) => console.error('Error fetching datasets:', err))
      .finally(() => setLoading(false));
  }, []);

  const filtered = datasets.filter(
    (d) =>
      d.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      d.dataset_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (d.task && d.task.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-6">
      {/* Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <Database className="w-5 h-5 text-cyan-400" />
            <h1 className="text-2xl font-bold text-white">Dataset Explorer & Metadata Inspection</h1>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Examine pre-extracted tensor archives, raw/epoch FIF files, channel counts, sampling frequencies, and class distributions.
          </p>
        </div>

        {/* Search */}
        <div className="relative w-full md:w-72">
          <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-500" />
          <input
            type="text"
            placeholder="Search dataset or paradigm..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Dataset Table / List */}
        <div className="lg:col-span-2 bg-slate-900/80 border border-slate-800 rounded-xl overflow-hidden shadow-sm flex flex-col">
          <div className="px-5 py-3 border-b border-slate-800 bg-slate-950/40 flex justify-between items-center text-xs">
            <span className="font-semibold text-slate-300">Cataloged Datasets ({filtered.length})</span>
            <span className="text-slate-500 font-mono text-[11px]">Storage: Drive D:\</span>
          </div>

          <div className="overflow-x-auto max-h-[550px] overflow-y-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/80 text-slate-400 border-b border-slate-800 sticky top-0">
                <tr>
                  <th className="px-4 py-2.5 font-semibold">Dataset / Task</th>
                  <th className="px-4 py-2.5 font-semibold">Format</th>
                  <th className="px-4 py-2.5 font-semibold">Channels</th>
                  <th className="px-4 py-2.5 font-semibold">Sampling</th>
                  <th className="px-4 py-2.5 font-semibold">Size</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {filtered.map((d) => {
                  const isSelected = selectedDataset?.dataset_id === d.dataset_id;
                  return (
                    <tr
                      key={d.dataset_id}
                      onClick={() => setSelectedDataset(d)}
                      className={`cursor-pointer transition-colors ${
                        isSelected
                          ? 'bg-cyan-500/10 text-cyan-300 font-medium'
                          : 'hover:bg-slate-800/40 text-slate-300'
                      }`}
                    >
                      <td className="px-4 py-2.5">
                        <div className="font-semibold">{d.dataset_id}</div>
                        <div className="text-[10px] text-slate-500">{d.task || 'ERP Paradigm'}</div>
                      </td>
                      <td className="px-4 py-2.5">
                        <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono text-[10px] border border-slate-700">
                          {d.file_type}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 font-mono">{d.num_channels} ch</td>
                      <td className="px-4 py-2.5 font-mono">{d.sampling_rate} Hz</td>
                      <td className="px-4 py-2.5 font-mono text-slate-400">{d.size_mb} MB</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Selected Dataset Detail Panel */}
        <div className="space-y-4">
          {selectedDataset ? (
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-sm space-y-4">
              <div className="border-b border-slate-800 pb-3">
                <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider">Dataset Inspector</span>
                <h3 className="text-base font-bold text-white mt-0.5 truncate">{selectedDataset.name}</h3>
                <p className="text-[11px] text-slate-400 font-mono break-all mt-1">{selectedDataset.path}</p>
              </div>

              {/* Warnings Banner */}
              {selectedDataset.warnings && selectedDataset.warnings.length > 0 && (
                <div className="p-3 bg-amber-950/30 border border-amber-500/30 rounded-lg text-amber-300 text-xs flex items-start space-x-2">
                  <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                  <span>{selectedDataset.warnings[0]}</span>
                </div>
              )}

              {/* Properties Grid */}
              <div className="space-y-2 text-xs">
                <div className="flex justify-between py-1 border-b border-slate-800/60">
                  <span className="text-slate-400">Total Channels</span>
                  <span className="font-bold text-slate-200">{selectedDataset.num_channels} Electrodes</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/60">
                  <span className="text-slate-400">Sampling Rate</span>
                  <span className="font-mono text-slate-200">{selectedDataset.sampling_rate} Hz</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/60">
                  <span className="text-slate-400">Epoch Time Window</span>
                  <span className="font-mono text-slate-200">-200 ms to +800 ms (1001 samples)</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/60">
                  <span className="text-slate-400">Classes</span>
                  <span className="font-bold text-cyan-400">2 (Binary ERP)</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800/60">
                  <span className="text-slate-400">Total Size</span>
                  <span className="font-mono text-slate-300">{selectedDataset.size_mb} MB</span>
                </div>
              </div>

              {/* Channel Tag Cloud */}
              <div className="space-y-1.5">
                <span className="text-[11px] font-semibold text-slate-400">Electrode Channel Montage:</span>
                <div className="flex flex-wrap gap-1 max-h-32 overflow-y-auto p-1.5 bg-slate-950 rounded border border-slate-800">
                  {selectedDataset.channel_names.map((ch) => (
                    <span
                      key={ch}
                      className="px-1.5 py-0.5 bg-slate-800 text-slate-300 rounded font-mono text-[10px]"
                    >
                      {ch}
                    </span>
                  ))}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="pt-2 space-y-2">
                <Link
                  to="/preprocessing"
                  className="w-full flex items-center justify-center space-x-2 py-2 px-3 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs transition"
                >
                  <span>Preview Preprocessing</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </Link>

                <Link
                  to="/inference"
                  className="w-full flex items-center justify-center space-x-2 py-2 px-3 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold text-xs border border-slate-700 transition"
                >
                  <span>Run Inference on this Dataset</span>
                </Link>
              </div>
            </div>
          ) : (
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-8 text-center text-slate-500 text-xs">
              <Database className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <span>Select a dataset from the table to inspect details.</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
