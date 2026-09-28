import type { DataSnapshot } from '@/lib/types';

/**
 * Unobtrusive footer showing how fresh each data source behind the
 * predictions is, from /model/stats' data_snapshot block. A null date
 * (backend couldn't read that manifest) drops its segment rather than
 * rendering "null". graph_status 'empty' adds a warning line above it --
 * 'unknown' deliberately doesn't, since the backend can't tell either way.
 */
export default function DataFreshness({ snapshot }: { snapshot: DataSnapshot }) {
  const segments = [
    snapshot.gtfs_static_snapshot && `Schedule data as of ${snapshot.gtfs_static_snapshot}`,
    snapshot.model_trained_through && `Model trained through ${snapshot.model_trained_through}`,
    snapshot.features_through && `Features updated through ${snapshot.features_through}`,
  ].filter(Boolean);

  return (
    <footer
      style={{
        marginTop: '3rem',
        paddingTop: '1rem',
        borderTop: '1px solid #ddd',
        textAlign: 'center',
        fontSize: '0.8em',
        color: '#666',
      }}
    >
      {snapshot.graph_status === 'empty' && (
        <p style={{ color: '#c04a2b', margin: '0 0 0.5rem' }}>
          Route data is being refreshed. Predictions unavailable right now.
        </p>
      )}
      {segments.length > 0 && <p style={{ margin: 0 }}>{segments.join(' · ')}</p>}
    </footer>
  );
}
