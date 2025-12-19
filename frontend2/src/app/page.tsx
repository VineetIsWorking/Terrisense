"use client";

import React, { useState } from 'react';
import { 
  UploadCloud, 
  Map, 
  FileSpreadsheet, 
  Plus, 
  Trash2, 
  CheckCircle, 
  AlertCircle,
  Loader2,
  Download
} from 'lucide-react';
import axios from 'axios';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

 function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- TYPES ---
type WeightRow = {
  id: number;
  column: string;
  weight: number;
};

// --- COMPONENTS ---

export default function Dashboard() {
  // State
  const [file, setFile] = useState<File | null>(null);
  const [numClusters, setNumClusters] = useState<number>(50);
  const [weights, setWeights] = useState<WeightRow[]>([{ id: 1, column: "", weight: 100 }]);
  
  const [loadingMap, setLoadingMap] = useState(false);
  const [loadingExcel, setLoadingExcel] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // Handlers
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const addWeightRow = () => {
    const newId = weights.length > 0 ? Math.max(...weights.map(w => w.id)) + 1 : 1;
    setWeights([...weights, { id: newId, column: "", weight: 0 }]);
  };

  const removeWeightRow = (id: number) => {
    setWeights(weights.filter(w => w.id !== id));
  };

  const updateWeight = (id: number, field: 'column' | 'weight', value: string | number) => {
    setWeights(weights.map(w => 
      w.id === id ? { ...w, [field]: value } : w
    ));
  };

  // --- API SUBMISSION ---
  const handleSubmit = async (endpoint: 'map' | 'excel') => {
    if (!file) {
      setError("Please upload a file first.");
      return;
    }
    
    // Validate Weights
    const totalWeight = weights.reduce((sum, w) => sum + Number(w.weight), 0);
    const validConfig = weights.filter(w => w.column.trim() !== "");
    
    if (validConfig.length === 0) {
      setError("Please define at least one column for weighting.");
      return;
    }

    // 1. Prepare JSON Config for API
    const configObject: Record<string, number> = {};
    validConfig.forEach(w => {
      configObject[w.column] = Number(w.weight);
    });

    // 2. Build Form Data
    const formData = new FormData();
    formData.append("file", file);
    formData.append("num_clusters", numClusters.toString());
    formData.append("column_config", JSON.stringify(configObject));

    // 3. Send Request
const url = endpoint === 'map' 
      ? 'https://terrisense.onrender.com/optimize_map' 
      : 'https://terrisense.onrender.com/optimize_excel';    
    const setLoading = endpoint === 'map' ? setLoadingMap : setLoadingExcel;
    
    try {
      setLoading(true);
      setError(null);
      setSuccessMsg(null);

      const response = await axios.post(url, formData, {
        responseType: 'blob', // Important for file downloads
      });

      // 4. Handle Download
      const downloadUrl = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.setAttribute('download', endpoint === 'map' ? 'territory_maps.zip' : 'territory_analysis.xlsx');
      document.body.appendChild(link);
      link.click();
      link.remove();

      setSuccessMsg(`Successfully generated ${endpoint === 'map' ? 'Maps' : 'Excel Report'}`);
    } catch (err: any) {
      console.error(err);
      setError("Failed to process request. Check if the column names exist in your file.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      {/* HEADER */}
      <header className="bg-white border-b border-slate-200 px-8 py-2 sticky top-0 z-10">
  <div className="max-w-7xl mx-auto flex items-center justify-between">
    
    {/* LEFT SIDE: Logo */}
    <div className="p-0 rounded-lg">
      <img 
        src="/images/T_logo.png" 
        alt="logo" 
        className="h-20 w-auto" 
      />
    </div>


  </div>
</header>

      {/* MAIN CONTENT */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* LEFT COLUMN: CONFIGURATION */}
          <div className="lg:col-span-7 space-y-6">
            
            {/* 1. File Upload Card */}
            <section className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-slate-100 text-xs text-slate-600">1</span>
                Upload Data
              </h2>
              
              <div className="relative group">
                <input 
                  type="file" 
                  accept=".csv, .xlsx"
                  onChange={handleFileChange}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                />
                <div className={cn(
                  "border-2 border-dashed rounded-lg p-8 text-center transition-all",
                  file ? "border-green-500 bg-green-50/50" : "border-slate-300 hover:border-blue-400 hover:bg-slate-50"
                )}>
                  <div className="flex flex-col items-center justify-center gap-2">
                    {file ? (
                      <>
                        <CheckCircle className="w-10 h-10 text-green-500" />
                        <p className="font-medium text-green-700">{file.name}</p>
                        <p className="text-xs text-green-600">{(file.size / 1024).toFixed(0)} KB</p>
                      </>
                    ) : (
                      <>
                        <UploadCloud className="w-10 h-10 text-slate-400 group-hover:text-blue-500 transition-colors" />
                        <p className="font-medium text-slate-600">Click or Drag Excel/CSV file here</p>
                        <p className="text-xs text-slate-400">Must contain Zip Code column</p>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </section>

            {/* 2. Parameters Card */}
            <section className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
              <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-slate-100 text-xs text-slate-600">2</span>
                Configuration
              </h2>

              {/* Territory Count */}
              <div className="mb-8">
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Target Number of Territories (K)
                </label>
                <div className="flex items-center gap-4">
                  <input 
                    type="range" 
                    min="2" 
                    max="200" 
                    value={numClusters} 
                    onChange={(e) => setNumClusters(Number(e.target.value))}
                    className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                  />
                  <input 
                    type="number" 
                    value={numClusters}
                    onChange={(e) => setNumClusters(Number(e.target.value))}
                    className="w-20 p-2 border border-slate-300 rounded-md text-center font-bold text-blue-600"
                  />
                </div>
                <p className="text-xs text-slate-500 mt-2">
                  We will also generate scenarios for <b>{Math.max(2, numClusters - 5)}</b> and <b>{numClusters + 5}</b> territories.
                </p>
              </div>

              <div className="h-px bg-slate-100 my-6" />

              {/* Dynamic Weights */}
              <div>
                <div className="flex items-center justify-between mb-4">
                  <label className="block text-sm font-medium text-slate-700">
                    Optimization Logic (Weighted Columns)
                  </label>
                  <button 
                    onClick={addWeightRow}
                    className="text-xs flex items-center gap-1 text-blue-600 font-medium hover:text-blue-800"
                  >
                    <Plus className="w-3 h-3" /> Add Column
                  </button>
                </div>

                <div className="space-y-3">
                  {weights.map((row) => (
                    <div key={row.id} className="flex gap-3 items-center animate-in fade-in slide-in-from-left-4 duration-300">
                      <div className="flex-1">
                        <input 
                          type="text" 
                          placeholder="Exact Column Name (e.g. Sales)" 
                          value={row.column}
                          onChange={(e) => updateWeight(row.id, 'column', e.target.value)}
                          className="w-full p-2 text-sm border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 outline-none"
                        />
                      </div>
                      <div className="w-24 relative">
                        <input 
                          type="number" 
                          placeholder="%" 
                          value={row.weight}
                          onChange={(e) => updateWeight(row.id, 'weight', e.target.value)}
                          className="w-full p-2 text-sm border border-slate-300 rounded-md focus:ring-2 focus:ring-blue-500 outline-none pr-6"
                        />
                        <span className="absolute right-2 top-2 text-slate-400 text-sm">%</span>
                      </div>
                      <button 
                        onClick={() => removeWeightRow(row.id)}
                        className="p-2 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-md transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-slate-400 mt-3">
                  * Ensure column names match your Excel file header exactly.
                </p>
              </div>
            </section>
          </div>

          {/* RIGHT COLUMN: ACTIONS & STATUS */}
          <div className="lg:col-span-5 space-y-6">
            
            {/* Status Panel */}
            <div className="bg-slate-900 text-white rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-medium mb-4">Ready to Optimize</h3>
              
              <div className="space-y-4">
                <div className="flex justify-between items-center text-sm text-slate-300 border-b border-slate-700 pb-2">
                  <span>File Status</span>
                  <span className={file ? "text-green-400 font-medium" : "text-slate-500"}>
                    {file ? "Ready" : "Waiting"}
                  </span>
                </div>
                <div className="flex justify-between items-center text-sm text-slate-300 border-b border-slate-700 pb-2">
                  <span>Logic</span>
                  <span className="text-white">
                     {weights.length} Criteria Defined
                  </span>
                </div>
                <div className="flex justify-between items-center text-sm text-slate-300 pb-2">
                  <span>Territories</span>
                  <span className="text-white font-mono bg-slate-800 px-2 py-1 rounded">
                    K = {numClusters} (±5)
                  </span>
                </div>
              </div>

              {/* Error / Success Messages */}
              {error && (
                <div className="mt-6 p-3 bg-red-500/10 border border-red-500/20 rounded-lg flex items-start gap-2">
                  <AlertCircle className="w-5 h-5 text-red-500 shrink-0" />
                  <p className="text-xs text-red-200">{error}</p>
                </div>
              )}
              
              {successMsg && (
                <div className="mt-6 p-3 bg-green-500/10 border border-green-500/20 rounded-lg flex items-start gap-2">
                  <CheckCircle className="w-5 h-5 text-green-500 shrink-0" />
                  <p className="text-xs text-green-200">{successMsg}</p>
                </div>
              )}
            </div>

            {/* Action Buttons */}
            <div className="grid gap-4">
              <button
                onClick={() => handleSubmit('map')}
                disabled={loadingMap || loadingExcel || !file}
                className="group relative w-full flex items-center justify-center gap-3 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white py-4 rounded-xl font-semibold shadow-md transition-all active:scale-[0.98]"
              >
                {loadingMap ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Map className="w-5 h-5 group-hover:scale-110 transition-transform" />
                )}
                Generate Visual Maps (ZIP)
              </button>

              <button
                onClick={() => handleSubmit('excel')}
                disabled={loadingMap || loadingExcel || !file}
                className="group w-full flex items-center justify-center gap-3 bg-white hover:bg-slate-50 border-2 border-slate-200 text-slate-700 py-4 rounded-xl font-semibold transition-all active:scale-[0.98]"
              >
                {loadingExcel ? (
                  <Loader2 className="w-5 h-5 animate-spin text-slate-600" />
                ) : (
                  <FileSpreadsheet className="w-5 h-5 text-green-600 group-hover:scale-110 transition-transform" />
                )}
                Download Excel Analysis
              </button>
            </div>

            {/* Helper Text */}
            <div className="bg-blue-50 border border-blue-100 p-4 rounded-lg text-xs text-blue-800 leading-relaxed">
              <strong>Tip:</strong> The 3-Scenario analysis creates "Original", "Original - 5", and "Original + 5" clusters. Download the Visual Map to see the geographic spread, then use the Excel sheet for detailed zip-level assignments.
            </div>

          </div>
        </div>
      </main>
    </div>
  );
}