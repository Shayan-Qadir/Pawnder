import React, { useEffect, useRef, useState } from 'react';
import { Timer, AlertTriangle } from 'lucide-react';

interface ClockWidgetProps {
  initialSeconds: number;
  isActive: boolean;
  onTimeOut: () => void;
  label: string;
  /** Change this to restart the clock without changing initialSeconds. */
  resetKey?: number;
}

export const ClockWidget: React.FC<ClockWidgetProps> = ({
  initialSeconds,
  isActive,
  onTimeOut,
  label,
  resetKey = 0,
}) => {
  const [seconds, setSeconds] = useState(initialSeconds);
  const secondsRef = useRef(initialSeconds);
  const timedOutRef = useRef(false);

  useEffect(() => {
    secondsRef.current = initialSeconds;
    timedOutRef.current = false;
    setSeconds(initialSeconds);
  }, [initialSeconds, resetKey]);

  useEffect(() => {
    if (!isActive) return;

    // Keep the countdown in a ref so the interval is not recreated on every tick.
    const interval = window.setInterval(() => {
      secondsRef.current = Math.max(0, secondsRef.current - 1);
      setSeconds(secondsRef.current);
    }, 1000);

    return () => window.clearInterval(interval);
  }, [isActive]);

  // Notify the parent from an effect, never from inside a state updater.
  useEffect(() => {
    if (isActive && seconds <= 0 && !timedOutRef.current) {
      timedOutRef.current = true;
      onTimeOut();
    }
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
