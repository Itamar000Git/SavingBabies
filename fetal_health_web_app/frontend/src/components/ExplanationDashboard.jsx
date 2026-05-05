import ExplanationPanel from './ExplanationPanel'
import GroundTruthPanel from './GroundTruthPanel'

function DashSection({ title, children }) {
  return (
    <div className="dash-section">
      <h3 className="dash-section-title">{title}</h3>
      <div className="dash-section-body">{children}</div>
    </div>
  )
}

function DashField({ label, value }) {
  const display = value == null ? 'Not available' : String(value)
  return (
    <div className="dash-field">
      <span className="dash-label">{label}</span>
      <span className={`dash-value${value == null ? ' dash-value--na' : ''}`}>{display}</span>
    </div>
  )
}

export default function ExplanationDashboard({
  explanation,
  signalFeatures,
  fhrEvents,
  metadata,
  groundTruth,
  prediction,
}) {
  const accelCount = fhrEvents?.accelerations?.length ?? 0
  const decelCount = fhrEvents?.decelerations?.length ?? 0
  const baby = metadata?.baby
  const mother = metadata?.mother

  return (
    <div className="explanation-dashboard">
      <DashSection title="FHR Summary">
        <DashField
          label="Mean FHR"
          value={signalFeatures?.fhr_mean != null ? `${signalFeatures.fhr_mean} bpm` : null}
        />
        <DashField
          label="FHR Std Dev"
          value={signalFeatures?.fhr_std != null ? `${signalFeatures.fhr_std} bpm` : null}
        />
        <DashField
          label="Missing Signal"
          value={signalFeatures?.missing_signal_pct != null ? `${signalFeatures.missing_signal_pct}%` : null}
        />
        <DashField
          label="Duration"
          value={signalFeatures?.recording_duration_min != null
            ? `${signalFeatures.recording_duration_min} min`
            : null}
        />
      </DashSection>

      <DashSection title="Uterine Contractions & Events">
        <DashField
          label="UC Activity"
          value={signalFeatures?.uc_available != null
            ? (signalFeatures.uc_available ? 'Present' : 'Absent')
            : null}
        />
        <DashField label="Accelerations detected" value={String(accelCount)} />
        <DashField label="Decelerations detected" value={String(decelCount)} />
      </DashSection>

      {baby && (
        <DashSection title="Baby Details">
          <DashField label="Baby ID" value={baby.baby_id} />
          <DashField
            label="Gestational Age"
            value={baby.gestational_weeks != null ? `${baby.gestational_weeks} weeks` : null}
          />
          <DashField
            label="Weight"
            value={baby.weight_g != null ? `${baby.weight_g} g` : null}
          />
          <DashField label="Sex" value={baby.sex} />
          <DashField label="Apgar 1 min" value={baby.apgar1} />
          <DashField label="Apgar 5 min" value={baby.apgar5} />
        </DashSection>
      )}

      {mother && (
        <DashSection title="Mother Details">
          <DashField
            label="Age"
            value={mother.mother_age != null ? `${mother.mother_age} years` : null}
          />
          <DashField label="Gravidity" value={mother.gravidity} />
          <DashField label="Parity" value={mother.parity} />
          <DashField
            label="Diabetes"
            value={mother.diabetes != null ? (mother.diabetes ? 'Yes' : 'No') : null}
          />
          <DashField
            label="Hypertension"
            value={mother.hypertension != null ? (mother.hypertension ? 'Yes' : 'No') : null}
          />
          <DashField
            label="Preeclampsia"
            value={mother.preeclampsia != null ? (mother.preeclampsia ? 'Yes' : 'No') : null}
          />
        </DashSection>
      )}

      {explanation && (
        <DashSection title="Model Explanation">
          <ExplanationPanel explanation={explanation} />
        </DashSection>
      )}

      {groundTruth && <GroundTruthPanel groundTruth={groundTruth} prediction={prediction} />}
    </div>
  )
}
