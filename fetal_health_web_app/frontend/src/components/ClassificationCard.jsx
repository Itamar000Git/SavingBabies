import BabyVisual from './BabyVisual'
import RiskScorePanel from './RiskScorePanel'
import ReliabilityPanel from './ReliabilityPanel'

export default function ClassificationCard({ prediction, reliability }) {
  return (
    <div className="classification-card">
      <BabyVisual prediction={prediction} />
      {prediction?.risk_score != null
        ? <RiskScorePanel prediction={prediction} />
        : reliability && <ReliabilityPanel reliability={reliability} prediction={prediction} />
      }
    </div>
  )
}
