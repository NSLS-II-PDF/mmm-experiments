from pydantic import BaseModel

DEFAULT_NAME = "agent_suggestions"


class Suggestion(BaseModel):
    ask_uid: str  # UID from the agent ask message
    plan_name: str
    plan_args: list = []
    plan_kwargs: dict = {}


class AdjudicatorMsg(BaseModel):
    agent_name: str
    suggestions_uid: str
    suggestions: dict[str, list[Suggestion]]  # TLA: list


if __name__ == "__main__":
    """Example main to show serializing capabilities"""
    import msgpack

    suggestion = Suggestion(ask_uid="123", plan_name="test_plan", plan_args=[1, 3], plan_kwargs={"md": {}})
    msg = AdjudicatorMsg(
        agent_name="aardvark",
        uid="456",
        suggestions={
            "pdf": [
                suggestion,
                suggestion,
            ],
            "bmm": [
                suggestion,
            ],
        },
    )
    print(msg)
    s = msgpack.dumps(msg.dict())
    new_msg = AdjudicatorMsg(**msgpack.loads(s))
    print(new_msg)
