import React from 'react';
import { UserProfile } from '../services/api';
import { Flame, ShieldAlert, Award, RefreshCw } from 'lucide-react';

interface WeaknessSummaryProps {
  profile: UserProfile | null;
  onRefresh: () => void;
  isLoading: boolean;
}

export const WeaknessSummary: React.FC<WeaknessSummaryProps> = ({
  profile,
  onRefresh,
  isLoading,
}) => {
  if (!profile) return null;

  return (
    <div className="bg-[#262421] border border-[#363431] rounded-xl p-5 shadow-xl">
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-[#363431]">
        <div>
          <h3 className="text-lg font-bold text-white flex items-center gap-2">
            <ShieldAlert className="w-5 h-5 text-amber-500" />
            Performance & Weakness Audit
          </h3>
          <p className="text-xs text-stone-400">
            Player: <span className="text-white font-medium">{profile.username}</span> | 10+0 Rapid
          </p>
        </div>
        <button
          onClick={onRefresh}
          disabled={isLoading}
          className="p-2 text-stone-400 hover:text-white bg-[#1e1c19] hover:bg-[#363431] rounded-lg transition"
          title="Re-analyze profile"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-3 gap-3 mb-5">
        <div className="bg-[#1e1c19] p-3 rounded-lg border border-[#363431]">
          <span className="text-xs text-stone-400 block">Games Audited</span>
          <span className="text-xl font-bold text-white">{profile.total_games}</span>
        </div>
        <div className="bg-[#1e1c19] p-3 rounded-lg border border-[#363431]">
          <span className="text-xs text-stone-400 block">Losses Analyzed</span>
          <span className="text-xl font-bold text-amber-400">{profile.losses_analyzed}</span>
        </div>
        <div className="bg-[#1e1c19] p-3 rounded-lg border border-[#363431]">
          <span className="text-xs text-stone-400 block">Blunder Stretch</span>
          <span className="text-xl font-bold text-red-400">{profile.blunders_extracted}</span>
        </div>
      </div>

      {/* Identified Weaknesses */}
      <div className="mb-4">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-stone-400 mb-2 flex items-center gap-1.5">
          <Flame className="w-4 h-4 text-red-500" />
          Model Diagnosed Weaknesses
        </h4>
        <ul className="space-y-2">
          {profile.weaknesses.map((w, idx) => (
            <li
              key={idx}
              className="text-sm bg-red-950/30 border border-red-900/40 text-red-200 px-3 py-2 rounded-lg flex items-start gap-2"
            >
              <span className="text-red-400 font-bold">•</span>
              <span>{w}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Sample Blunders Extracted */}
      {profile.sample_blunders && profile.sample_blunders.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold uppercase tracking-wider text-stone-400 mb-2 flex items-center gap-1.5">
            <Award className="w-4 h-4 text-[#81b64c]" />
            Recent Deterioration Points
          </h4>
          <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
            {profile.sample_blunders.map((b, idx) => (
              <div
                key={idx}
                className="text-xs bg-[#1e1c19] border border-[#363431] p-2.5 rounded-lg flex justify-between items-center"
              >
                <div>
                  <span className="font-semibold text-white">Move {Math.floor((b.ply || 20) / 2)}</span>:
                  <span className="text-stone-300 ml-1">{b.reason || 'Tactical slip'}</span>
                </div>
                <div className="text-right">
                  <span className="text-red-400 font-mono">-{b.cp_drop || 150} cp</span>
                  <span className="text-stone-500 block text-[10px]">
                    {Math.round(b.clock_remaining || 45)}s left
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
