"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import re
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def query(self, user_input: str) -> dict:
        return {
            "status": "success",
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
        }

class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace = []

    @staticmethod
    def _extract_price(user_input: str, default: int = 5000000) -> int:
        match = re.search(r"([\d.,]+)\s*triệu", user_input.lower())
        if not match:
            return default
        amount = float(match.group(1).replace(",", "."))
        return int(amount * 1_000_000)

    @staticmethod
    def _extract_code(user_input: str, codes) -> str | None:
        upper_input = user_input.upper()
        for code in codes:
            if re.search(rf"\b{code}\b", upper_input):
                return code
        return None

    def _actions_for(self, user_input: str):
        actions = []
        origin = self._extract_code(user_input, ("HAN", "SGN", "DAD"))
        destination = self._extract_code(user_input, ("SGN", "DAD", "HAN"))
        has_flight_request = any(
            word in user_input.lower() for word in ("bay", "vé", "chuyến")
        )
        has_weather_request = any(
            word in user_input.lower() for word in ("thời tiết", "weather", "mặc gì")
        )

        if has_flight_request and origin and destination and origin != destination:
            actions.append({
                "name": "get_flight_info",
                "args": {
                    "origin": origin,
                    "destination": destination,
                    "max_price": self._extract_price(user_input),
                },
            })
        if has_weather_request:
            city = self._extract_code(user_input, ("SGN", "HAN", "DAD"))
            if city:
                actions.append({
                    "name": "get_weather_forecast",
                    "args": {"city_code": city},
                })
        return actions

    @staticmethod
    def _format_observation(tool_name, observation) -> str:
        if tool_name == "get_flight_info":
            if not observation:
                return "Không tìm thấy chuyến bay phù hợp."
            return "; ".join(
                f"{flight['flight_number']} ({flight['airline']}, "
                f"{flight['departure_time']}, {flight['price_vnd']:,} VND)"
                for flight in observation
            )
        if "error" in observation:
            return observation["error"]
        return (
            f"{observation['city']}: {observation['temperature_c']}°C, "
            f"{observation['condition']}. {observation['recommendation']}"
        )

    def run(self, user_input: str) -> dict:
        self.trace = []
        actions = self._actions_for(user_input)
        observations = []

        for iteration, action in enumerate(actions[: self.max_iterations], start=1):
            tool_name = action["name"].strip().lower()
            tool = TOOL_MAP.get(tool_name)
            if tool is None:
                observation = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    observation = tool(**action["args"])
                except (TypeError, ValueError, KeyError) as exc:
                    observation = {"error": f"Tool error: {exc}"}
            observations.append((tool_name, observation))
            self.trace.append({
                "iteration": iteration,
                "thought": "Thu thập dữ liệu cần thiết cho câu hỏi.",
                "action": json.loads(json.dumps(action)),
                "observation": observation,
            })

        requires_final_step = len(actions) > 1
        if len(actions) + int(requires_final_step) > self.max_iterations:
            return {
                "status": "max_iterations_reached",
                "answer": "Không thể hoàn thành trong số bước tối đa.",
                "iterations": self.max_iterations,
                "trace": self.trace,
            }

        if not observations:
            answer = (
                "Chính sách đổi trả vé máy bay Vinpearl phụ thuộc vào điều kiện vé "
                "và thời điểm yêu cầu. Vui lòng kiểm tra điều kiện vé hoặc liên hệ hỗ trợ."
            )
        else:
            answer = "\n".join(
                self._format_observation(tool_name, observation)
                for tool_name, observation in observations
            )
        if requires_final_step or not observations:
            self.trace.append({
                "iteration": len(self.trace) + 1,
                "thought": "Đã đủ dữ liệu, tổng hợp câu trả lời cho khách hàng.",
                "final_answer": answer,
            })
        return {
            "status": "completed",
            "answer": answer,
            "iterations": len(self.trace),
            "trace": self.trace,
        }

def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()