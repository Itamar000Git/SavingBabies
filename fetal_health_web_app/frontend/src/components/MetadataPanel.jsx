function MetaField({ label, value }) {
  const display = value == null ? 'Not available' : String(value)
  return (
    <div className="meta-field">
      <span className="meta-label">{label}</span>
      <span className={`meta-value${value == null ? ' not-available' : ''}`}>{display}</span>
    </div>
  )
}

function MetaSection({ title, children }) {
  return (
    <div className="meta-section">
      <h3 className="meta-section-title">{title}</h3>
      {children}
    </div>
  )
}

export default function MetadataPanel({ metadata }) {
  if (!metadata) {
    return (
      <div className="metadata-panel">
        <p className="meta-placeholder">Upload a recording to see patient details</p>
      </div>
    )
  }

  const { baby, mother, medical } = metadata

  return (
    <div className="metadata-panel">
      <MetaSection title="Baby Details">
        <MetaField label="Baby ID" value={baby.baby_id} />
        <MetaField label="Gestational Age" value={baby.gestational_weeks != null ? `${baby.gestational_weeks} weeks` : null} />
        <MetaField label="Weight" value={baby.weight_g != null ? `${baby.weight_g} g` : null} />
        <MetaField label="Sex" value={baby.sex} />
        <MetaField label="Apgar 1 min" value={baby.apgar1} />
        <MetaField label="Apgar 5 min" value={baby.apgar5} />
      </MetaSection>

      <MetaSection title="Mother Details">
        <MetaField label="Age" value={mother.mother_age != null ? `${mother.mother_age} years` : null} />
        <MetaField label="Gravidity" value={mother.gravidity} />
        <MetaField label="Parity" value={mother.parity} />
        <MetaField label="Diabetes" value={mother.diabetes != null ? (mother.diabetes ? 'Yes' : 'No') : null} />
        <MetaField label="Hypertension" value={mother.hypertension != null ? (mother.hypertension ? 'Yes' : 'No') : null} />
        <MetaField label="Preeclampsia" value={mother.preeclampsia != null ? (mother.preeclampsia ? 'Yes' : 'No') : null} />
      </MetaSection>

      <MetaSection title="Medical / Recording Status">
        <MetaField label="Duration" value={medical.recording_duration_min != null ? `${medical.recording_duration_min} min` : null} />
        <MetaField label="Missing Signal" value={medical.missing_signal_pct != null ? `${medical.missing_signal_pct}%` : null} />
        <MetaField label="FHR Mean" value={medical.fhr_mean != null ? `${medical.fhr_mean} bpm` : null} />
        <MetaField label="FHR Std Dev" value={medical.fhr_std != null ? `${medical.fhr_std} bpm` : null} />
        <MetaField label="UC Activity" value={medical.uc_available != null ? (medical.uc_available ? 'Present' : 'Absent') : null} />
      </MetaSection>
    </div>
  )
}
