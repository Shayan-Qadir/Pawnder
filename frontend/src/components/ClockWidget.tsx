import React, { useEffect, useState } from 'react';
import { Timer, AlertTriangle } from 'lucide-react';

interface ClockWidgetProps {
  initialSeconds: number;
  isActive: boolean;
  onTimeOut: () => void;
  label: string;
}

export const ClockWidget: React.FC<ClockWidgetProps> = ({
  initialSeconds,
  isActive,
  onTimeOut,
  label,
}) => {
  const [seconds, setSeconds] = useState(initialSeconds);

  useEffect(() => {
    setSeconds(initialSeconds);
  }, [initialSeconds]);

  useEffect(() => {
    if (!isActive || seconds <= 0) return;

    const interval = setInterval(() => {
      setSeconds((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          onTimeOut();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [isActive, seconds, onTimeOut]);

  const minutes = Math.floor(seconds / 60);
  const remainingSecs = Math.floor(seconds % 60);
  const isScramble = seconds < 30;

  return (
    <div
      className={`flex items-center justify-between px-4 py-2.5 rounded-lg border transition-all ${
        isActive
          ? isScramble
            ? 'bg-red-950/60 border-red-500 animate-pulse text-red-200'
            : 'bg-[#262421] border-[#81b64c] text-white shadow-lg shadow-green-950/20'
          : 'bg-[#1e1c19] border-[#363431] text-stone-400'
      }`}
    >
      <div className="flex items-center gap-2">
        {isScramble && isActive ? (
          <AlertTriangle className="w-5 h-5 text-red-400 animate-bounce" />
        ) : (
          <Timer className={`w-5 h-5 ${isActive ? 'text-[#81b64c]' : 'text-stone-500'}`} />
        )}
        <span className="text-sm font-medium">{label}</span>
      </div>
      <div className="font-mono text-2xl font-bold tracking-wider">
        {String(minutes).padStart(2, '0')}:{String(remainingSecs).padStart(2, '0')}
      </div>
    </div>
  );
};
