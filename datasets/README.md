
### Template:

```json
[
{
    "utterance": "Topic phrase",
    "Slots":
    {
        "intention": "target: request, report, define, confirm, deny, prefer, etc.",
        "action": "action: provide, contain, update, approve, reject, etc.",
        "relation": "relation type between subject and object: contains, describes, depends_on, must_have, etc.",
        "modality": "possibility, ability, assumption...",
        "emotion": "emotional tone: neutral, like, dislike, frustration, satisfaction",
        "subject": "object: person or entity or entities that the action is directed at.",
        "object": "emotion: the emotional tone if expressed (e.g. neutral, positive, negative, frustration, satisfaction)"
    }
}
]
```


| English word       | Modality type                          | Example                            | Meaning (in English)                              |
| ------------------ | -------------------------------------- | ---------------------------------- | ------------------------------------------ |
| **may**            | **possibility / permission**           | “It may happen.”                   | expresses **possibility** or **permission**|
| **can**            | **ability / capability / possibility** | “It can be done.”                  | expresses **ability** or **potential**     |
| **should**         | **recommendation / obligation (weak)** | “You should check it.”             | expresses **advice** or **weak obligation**|
| **must**           | **necessity / obligation (strong)**    | “You must define the plan.”        | expresses **strong requirement** or **necessity**|
| **might**          | **hypothetical possibility**           | “It might fail.”                   | expresses **uncertain possibility** |
| **will**           | **certainty / future prediction**      | “It will start soon.”              | expresses **certainty** or **future intention**|
| **is expected to** | **expectation / assumption**           | “The team is expected to deliver.” | expresses **expectation** or **assumed outcome**|
