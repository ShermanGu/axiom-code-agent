from __future__ import annotations

import json
import unittest

from axiom_agent.planning.planner import Planner
from axiom_agent.providers.demo import DemoProvider


class PlannerCapabilityTests(unittest.TestCase):
    def test_planner_sees_compact_capabilities_without_tool_call_access(self) -> None:
        planner = Planner(DemoProvider())
        request = planner.build_request(
            "Summarize today's mail",
            tool_catalog=[
                {
                    "name": "mcp__mail__search_email",
                    "description": "Search the mailbox",
                    "arguments": ["query"],
                }
            ],
        )

        self.assertEqual(request.tools, [])
        prompt = str(request.input_items[0]["content"])
        self.assertIn("Available tool capabilities", prompt)
        self.assertIn("mcp__mail__search_email", prompt)

    def test_plan_preserves_capability_requirements_and_candidate_tools(self) -> None:
        planner = Planner(DemoProvider())
        plan = planner.parse_response(
            "Summarize today's mail",
            json.dumps(
                {
                    "strategy": "Read before summarizing.",
                    "steps": [
                        {
                            "id": "read_mail",
                            "title": "Read mail",
                            "description": "Collect today's messages.",
                            "depends_on": [],
                            "required_capabilities": ["read mailbox data"],
                            "candidate_tools": [
                                "mcp__mail__search_email",
                                "invented_tool",
                            ],
                        }
                    ],
                }
            ),
            available_tools={"mcp__mail__search_email"},
        )

        step = plan.steps[0]
        self.assertEqual(step.required_capabilities, ["read mailbox data"])
        self.assertEqual(step.candidate_tools, ["mcp__mail__search_email"])
        self.assertEqual(
            step.as_dict()["candidate_tools"], ["mcp__mail__search_email"]
        )


if __name__ == "__main__":
    unittest.main()
