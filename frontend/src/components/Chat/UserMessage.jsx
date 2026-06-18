export default function UserMessage({ content }) {
  return (
    <div className="message message--user">
      <div className="message__bubble message__bubble--user">
        <p className="message__text">{content}</p>
      </div>
    </div>
  )
}
