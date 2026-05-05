import { useMemo } from 'react'
import {
  ComposedChart,
  AreaChart,
  Area,
  Line,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'

function findClosestIdx(arr, val) {
  let best = 0
  let bestDist = Infinity
  for (let i = 0; i < arr.length; i++) {
    const d = Math.abs(arr[i] - val)
    if (d < bestDist) { bestDist = d; best = i }
  }
  return best
}

function AccelDot({ cx, cy }) {
  if (cx == null || cy == null) return null
  return <circle cx={cx} cy={cy} r={5} fill="#22c55e" stroke="#fff" strokeWidth={1.5} />
}

function DecelDot({ cx, cy, fill }) {
  if (cx == null || cy == null) return null
  return <circle cx={cx} cy={cy} r={5} fill={fill ?? '#ef4444'} stroke="#fff" strokeWidth={1.5} />
}

function subtypeColor(subtype) {
  if (subtype === 'prolonged') return '#7c3aed'
  if (subtype === 'gradual') return '#f97316'
  return '#ef4444' // variable
}

function FHRTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const fhrEntry = payload.find(p => p.dataKey === 'fhr')
  if (!fhrEntry) return null
  return (
    <div className="ctg-tooltip">
      <div className="ctg-tooltip-time">{Number(label).toFixed(1)} min</div>
      {fhrEntry.value != null && (
        <div>FHR: <strong>{fhrEntry.value} bpm</strong></div>
      )}
    </div>
  )
}

const xTickFmt = v => `${Number(v).toFixed(0)}m`

export default function CTGChartPanel({ signalData, fhrEvents }) {
  const fhrData = useMemo(() => {
    if (!signalData) return []
    const { time_min, fhr } = signalData
    return time_min.map((t, i) => ({ t, fhr: fhr[i] }))
  }, [signalData])

  const ucData = useMemo(() => {
    if (!signalData) return []
    return signalData.time_min.map((t, i) => ({ t, uc: signalData.uc[i] }))
  }, [signalData])

  const accelPoints = useMemo(() => {
    if (!fhrEvents || !signalData) return []
    return fhrEvents.accelerations.map(ev => {
      const idx = findClosestIdx(signalData.time_min, ev.peak_or_nadir_min)
      const fhrVal = signalData.fhr[idx]
      if (fhrVal == null) return null
      return { t: signalData.time_min[idx], fhr: fhrVal }
    }).filter(Boolean)
  }, [fhrEvents, signalData])

  const decelPoints = useMemo(() => {
    if (!fhrEvents || !signalData) return []
    return fhrEvents.decelerations.map(ev => {
      const idx = findClosestIdx(signalData.time_min, ev.peak_or_nadir_min)
      const fhrVal = signalData.fhr[idx]
      if (fhrVal == null) return null
      return { t: signalData.time_min[idx], fhr: fhrVal, subtype: ev.subtype }
    }).filter(Boolean)
  }, [fhrEvents, signalData])

  if (!signalData) return null

  const accelCount = fhrEvents?.accelerations?.length ?? 0
  const decelCount = fhrEvents?.decelerations?.length ?? 0

  return (
    <div className="ctg-chart-panel">
      <div className="ctg-chart-header">
        <h2 className="ctg-title">CTG Recording</h2>
        <div className="ctg-event-legend">
          <span className="ctg-legend-chip ctg-legend-accel">
            <span className="ctg-legend-dot" style={{ background: '#22c55e' }} />
            {accelCount} Acceleration{accelCount !== 1 ? 's' : ''}
          </span>
          <span className="ctg-legend-chip ctg-legend-decel">
            <span className="ctg-legend-dot" style={{ background: '#ef4444' }} />
            {decelCount} Deceleration{decelCount !== 1 ? 's' : ''}
          </span>
        </div>
      </div>

      <div className="ctg-chart-section">
        <div className="ctg-chart-label">FHR (bpm)</div>
        <ResponsiveContainer width="100%" height={200}>
          <ComposedChart data={fhrData} margin={{ top: 5, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="t"
              type="number"
              domain={['dataMin', 'dataMax']}
              tickFormatter={xTickFmt}
              tick={{ fontSize: 11 }}
              scale="linear"
            />
            <YAxis domain={[50, 220]} tick={{ fontSize: 11 }} width={36} />
            <Tooltip content={<FHRTooltip />} />
            <Line
              dataKey="fhr"
              stroke="#3b82f6"
              strokeWidth={1.5}
              dot={false}
              connectNulls={false}
              isAnimationActive={false}
            />
            {accelPoints.length > 0 && (
              <Scatter
                data={accelPoints}
                dataKey="fhr"
                shape={<AccelDot />}
                isAnimationActive={false}
                legendType="none"
              />
            )}
            {decelPoints.length > 0 && (
              <Scatter
                data={decelPoints}
                dataKey="fhr"
                shape={(props) => <DecelDot {...props} fill={subtypeColor(props.subtype)} />}
                isAnimationActive={false}
                legendType="none"
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="ctg-chart-section ctg-uc-section">
        <div className="ctg-chart-label">UC (TOCO)</div>
        <ResponsiveContainer width="100%" height={100}>
          <AreaChart data={ucData} margin={{ top: 5, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="t"
              type="number"
              domain={['dataMin', 'dataMax']}
              tickFormatter={xTickFmt}
              tick={{ fontSize: 11 }}
              scale="linear"
            />
            <YAxis tick={{ fontSize: 11 }} width={36} />
            <Area
              dataKey="uc"
              stroke="#f59e0b"
              fill="#fef3c7"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
