'use client';

import { motion } from 'framer-motion';
import { iconForMode } from '@/lib/modeIcons';
import type { Confidence, RouteOption } from '@/lib/types';

// Same color scheme as phase3/ui/formatting.py's CONFIDENCE_COLOR --
// reused verbatim, not reinvented (High/Medium/Low only; predict_delay()
// never emits anything else).
const CONFIDENCE_COLOR: Record<Confidence, string> = {
  High: '#3AA65B',
  Medium: '#E0A526',
  Low: '#8A8A8A',
};

function ConfidenceBadge({ confidence }: { confidence: Confidence }) {
  const color = CONFIDENCE_COLOR[confidence];
  return (
    <span
      style={{
        background: `${color}22`,
        color,
        padding: '2px 10px',
        borderRadius: 12,
        fontWeight: 600,
        fontSize: '0.75em',
        whiteSpace: 'nowrap',
      }}
    >
      {confidence}
    </span>
  );
}

/** 'h:MM AM/PM', no leading zero on the hour -- same convention as
 * phase3/ui/formatting.py's _format_time_ampm_short(). departure_time is
 * a naive ISO string (no timezone suffix), so `new Date(iso)` parses it
 * as local wall-clock time directly, matching what it represents.
 */
function formatHHMM(iso: string): string {
  const d = new Date(iso);
  const hours = d.getHours();
  const minutes = d.getMinutes();
  const period = hours < 12 ? 'AM' : 'PM';
  const hour12 = hours % 12 || 12;
  return `${hour12}:${minutes.toString().padStart(2, '0')} ${period}`;
}

/**
 * Animated hero visual for the single top-ranked /routes journey: a
 * horizontal line with one node per leg, each showing that leg's actual
 * mode icon (looked up generically -- no assumption about which modes or
 * how many legs a given journey has) and its own confidence badge, plus a
 * "Leave by" readout from the first leg's departure_time. Works for any
 * leg count, including a single direct leg (no connecting line rendered
 * when there's nothing to connect).
 */
export default function RouteHero({ route }: { route: RouteOption }) {
  const { legs, total_predicted_duration_minutes, transfer_count } = route;
  const firstLeg = legs[0];

  return (
    <div style={{ padding: '1.5rem', border: '1px solid #ddd', borderRadius: 12 }}>
      <motion.p
        initial={{ opacity: 0, y: -6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        style={{ fontSize: '1.1em', fontWeight: 600, margin: 0 }}
      >
        🕒 Leave by {formatHHMM(firstLeg.departure_time)}
      </motion.p>
      <p style={{ color: '#666', margin: '0.25rem 0 1.5rem' }}>
        {total_predicted_duration_minutes} min ·{' '}
        {transfer_count === 0 ? 'direct' : `${transfer_count} transfer${transfer_count > 1 ? 's' : ''}`}
      </p>

      <div style={{ display: 'flex', alignItems: 'flex-start' }}>
        {legs.map((leg, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'flex-start', flex: i < legs.length - 1 ? 1 : undefined }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', minWidth: 64 }}>
              <motion.div
                initial={{ opacity: 0, scale: 0.4 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.35, duration: 0.35, ease: 'easeOut' }}
                style={{
                  width: 48,
                  height: 48,
                  borderRadius: '50%',
                  background: '#f3f4f6',
                  border: '2px solid #d1d5db',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '1.4em',
                }}
                title={`mode: ${leg.mode}`}
              >
                {iconForMode(leg.mode)}
              </motion.div>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: i * 0.35 + 0.15, duration: 0.3 }}
                style={{ textAlign: 'center', marginTop: 6 }}
              >
                <div style={{ fontWeight: 600, fontSize: '0.9em' }}>{leg.route_short_name}</div>
                <div style={{ marginTop: 4 }}>
                  <ConfidenceBadge confidence={leg.confidence} />
                </div>
              </motion.div>
            </div>

            {i < legs.length - 1 && (
              <motion.div
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ delay: i * 0.35 + 0.35, duration: 0.35, ease: 'easeOut' }}
                style={{
                  flex: 1,
                  height: 3,
                  background: '#d1d5db',
                  marginTop: 23,
                  transformOrigin: 'left',
                }}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
