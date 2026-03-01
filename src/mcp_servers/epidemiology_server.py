"""
ORACLE — Epidemiology MCP Server
Outbreak data, case counts, vaccination rates, and spread forecasting tools.
"""

import json
import logging
import random
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EpidemiologyMCPServer:
    """
    MCP Server for epidemiological data and forecasting.

    Tools:
    - get_outbreak_data: Get current outbreak status by region
    - get_case_counts: Get time-series case data
    - get_vaccination_rates: Get vaccination coverage data
    - forecast_spread: Forecast variant spread trajectory
    - get_variant_prevalence: Get variant prevalence over time
    """

    SERVER_NAME = "oracle-epidemiology-server"
    SERVER_VERSION = "1.0.0"

    REGIONS = ["North America", "Europe", "Asia", "South America", "Africa", "Oceania"]
    COUNTRIES = {
        "North America": ["United States", "Canada", "Mexico"],
        "Europe": ["United Kingdom", "Germany", "France", "Italy", "Spain", "Netherlands", "Denmark"],
        "Asia": ["India", "Japan", "South Korea", "China", "Thailand", "Indonesia"],
        "South America": ["Brazil", "Argentina"],
        "Africa": ["South Africa", "Nigeria", "Kenya"],
        "Oceania": ["Australia"],
    }

    def __init__(self):
        self._data_cache = {}
        logger.info(f"Initialized {self.SERVER_NAME} v{self.SERVER_VERSION}")

    def get_server_info(self) -> dict:
        return {
            "name": self.SERVER_NAME,
            "version": self.SERVER_VERSION,
            "protocol": "MCP/1.0",
            "capabilities": {"tools": True, "resources": True},
            "tools": self.list_tools(),
        }

    def list_tools(self) -> list[dict]:
        return [
            {
                "name": "get_outbreak_data",
                "description": "Get current outbreak status and alert level by region",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "description": "Region name or 'global'"},
                    },
                },
            },
            {
                "name": "get_case_counts",
                "description": "Get time-series COVID-19 case data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "country": {"type": "string"},
                        "days": {"type": "integer", "default": 30},
                    },
                },
            },
            {
                "name": "get_vaccination_rates",
                "description": "Get vaccination coverage by country/region",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "region": {"type": "string", "default": "global"},
                    },
                },
            },
            {
                "name": "forecast_spread",
                "description": "Forecast variant spread trajectory using SIR model",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "variant": {"type": "string", "description": "Variant lineage name"},
                        "region": {"type": "string", "default": "global"},
                        "days": {"type": "integer", "default": 90},
                    },
                    "required": ["variant"],
                },
            },
            {
                "name": "get_variant_prevalence",
                "description": "Get variant prevalence percentages over time",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "weeks": {"type": "integer", "default": 12},
                    },
                },
            },
        ]

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        handlers = {
            "get_outbreak_data": self._get_outbreak_data,
            "get_case_counts": self._get_case_counts,
            "get_vaccination_rates": self._get_vaccination_rates,
            "forecast_spread": self._forecast_spread,
            "get_variant_prevalence": self._get_variant_prevalence,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            result = handler(**arguments)
            return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}
        except Exception as e:
            return {"error": str(e)}

    def _get_outbreak_data(self, region: str = "global") -> dict:
        """Generate realistic outbreak data."""
        rng = random.Random(hash(region))

        if region == "global":
            regions_data = {}
            for r in self.REGIONS:
                regions_data[r] = self._generate_region_outbreak(r, rng)
            return {
                "report_date": datetime.now().isoformat(),
                "global_status": "ELEVATED",
                "regions": regions_data,
                "global_weekly_cases": sum(r["weekly_cases"] for r in regions_data.values()),
                "dominant_variant": "JN.1",
                "emerging_variants": ["JN.1.1", "KP.2", "KP.3"],
            }

        return self._generate_region_outbreak(region, rng)

    def _generate_region_outbreak(self, region: str, rng: random.Random) -> dict:
        return {
            "region": region,
            "alert_level": rng.choice(["LOW", "MODERATE", "ELEVATED", "HIGH"]),
            "weekly_cases": rng.randint(5000, 500000),
            "weekly_deaths": rng.randint(50, 5000),
            "hospitalization_rate": round(rng.uniform(0.5, 5.0), 2),
            "positivity_rate": round(rng.uniform(2.0, 25.0), 2),
            "r_effective": round(rng.uniform(0.7, 2.5), 2),
            "dominant_variant": rng.choice(["JN.1", "XBB.1.5", "BA.5", "BA.2.86"]),
            "variant_growth_rate": round(rng.uniform(-0.05, 0.2), 4),
            "sequencing_coverage": round(rng.uniform(1.0, 15.0), 2),
        }

    def _get_case_counts(self, country: str = "United States", days: int = 30) -> dict:
        """Generate time-series case data."""
        rng = random.Random(hash(country))
        base_cases = rng.randint(1000, 50000)

        time_series = []
        today = datetime.now()
        for i in range(days):
            date = today - timedelta(days=days - i)
            daily_cases = max(0, int(base_cases * (1 + 0.02 * i) + rng.gauss(0, base_cases * 0.1)))
            daily_deaths = max(0, int(daily_cases * rng.uniform(0.005, 0.02)))

            time_series.append({
                "date": date.strftime("%Y-%m-%d"),
                "cases": daily_cases,
                "deaths": daily_deaths,
                "hospitalizations": int(daily_cases * rng.uniform(0.01, 0.05)),
                "tests": int(daily_cases * rng.uniform(3, 10)),
                "positivity_rate": round(rng.uniform(3.0, 20.0), 2),
            })

        return {
            "country": country,
            "period_days": days,
            "total_cases": sum(d["cases"] for d in time_series),
            "total_deaths": sum(d["deaths"] for d in time_series),
            "avg_daily_cases": round(sum(d["cases"] for d in time_series) / days),
            "trend": "INCREASING" if time_series[-1]["cases"] > time_series[0]["cases"] else "DECREASING",
            "time_series": time_series,
        }

    def _get_vaccination_rates(self, region: str = "global") -> dict:
        """Get vaccination coverage data."""
        rng = random.Random(hash(region))

        countries_data = []
        all_countries = []
        for r, cs in self.COUNTRIES.items():
            if region == "global" or r == region:
                all_countries.extend(cs)

        for country in all_countries:
            crng = random.Random(hash(country))
            countries_data.append({
                "country": country,
                "primary_series": round(crng.uniform(40, 95), 1),
                "first_booster": round(crng.uniform(20, 80), 1),
                "second_booster": round(crng.uniform(5, 50), 1),
                "bivalent_booster": round(crng.uniform(2, 35), 1),
                "updated_2024": round(crng.uniform(1, 25), 1),
                "total_doses_administered": crng.randint(10_000_000, 500_000_000),
            })

        return {
            "region": region,
            "report_date": datetime.now().strftime("%Y-%m-%d"),
            "countries": countries_data,
            "global_avg_primary": round(sum(c["primary_series"] for c in countries_data) / max(len(countries_data), 1), 1),
            "global_avg_booster": round(sum(c["first_booster"] for c in countries_data) / max(len(countries_data), 1), 1),
        }

    def _forecast_spread(self, variant: str, region: str = "global", days: int = 90) -> dict:
        """SIR model-based spread forecast."""
        rng = random.Random(hash(f"{variant}_{region}"))

        # SIR parameters
        population = rng.randint(50_000_000, 500_000_000)
        initial_infected = rng.randint(10_000, 1_000_000)
        r0 = rng.uniform(1.5, 4.0)
        gamma = 1 / 10  # Recovery rate (10 day avg illness)
        beta = r0 * gamma

        # Simplified SIR simulation
        S = population - initial_infected
        I = initial_infected
        R = 0
        dt = 1.0

        forecast = []
        for day in range(days):
            new_infections = beta * S * I / population * dt
            new_recoveries = gamma * I * dt

            S -= new_infections
            I += new_infections - new_recoveries
            R += new_recoveries

            S = max(0, S)
            I = max(0, I)

            if day % 7 == 0:  # Weekly data points
                forecast.append({
                    "week": day // 7 + 1,
                    "date": (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d"),
                    "estimated_cases": int(I),
                    "cumulative_cases": int(R),
                    "prevalence_pct": round(I / population * 100, 4),
                    "susceptible_pct": round(S / population * 100, 2),
                })

        peak_week = max(forecast, key=lambda x: x["estimated_cases"])

        return {
            "variant": variant,
            "region": region,
            "model": "SIR (Susceptible-Infected-Recovered)",
            "r0": round(r0, 2),
            "population": population,
            "initial_cases": initial_infected,
            "peak_cases": peak_week["estimated_cases"],
            "peak_week": peak_week["week"],
            "forecast_weeks": len(forecast),
            "weekly_forecast": forecast,
        }

    def _get_variant_prevalence(self, weeks: int = 12) -> dict:
        """Get variant prevalence over time."""
        rng = random.Random(42)
        variants = ["JN.1", "XBB.1.5", "BA.5", "BA.2.86", "EG.5", "HK.3", "Other"]

        prevalence_data = []
        today = datetime.now()

        # Simulate variant dynamics (logistic growth competition)
        for w in range(weeks):
            week_date = today - timedelta(weeks=weeks - w)
            proportions = {}
            remaining = 100.0

            for i, var in enumerate(variants[:-1]):
                # Sigmoid-like growth/decay for each variant
                t = w / weeks
                if var == "JN.1":
                    prop = 15 + 50 * (1 / (1 + 2.718 ** (-10 * (t - 0.4))))
                elif var == "XBB.1.5":
                    prop = 40 * (1 - t) ** 2
                elif var == "BA.5":
                    prop = 10 * max(0, 1 - 2 * t)
                elif var == "BA.2.86":
                    prop = 5 + 10 * t
                elif var == "EG.5":
                    prop = 15 * (1 - t)
                elif var == "HK.3":
                    prop = 3 + 8 * t
                else:
                    prop = 5

                prop = max(0, min(remaining, prop + rng.gauss(0, 2)))
                proportions[var] = round(prop, 1)
                remaining -= prop

            proportions["Other"] = round(max(0, remaining), 1)

            prevalence_data.append({
                "week": week_date.strftime("%Y-%m-%d"),
                "week_number": w + 1,
                "proportions": proportions,
            })

        return {
            "period_weeks": weeks,
            "variants_tracked": variants,
            "current_dominant": max(prevalence_data[-1]["proportions"],
                                     key=prevalence_data[-1]["proportions"].get),
            "fastest_growing": "JN.1",
            "prevalence_data": prevalence_data,
        }


def main():
    server = EpidemiologyMCPServer()
    print(json.dumps(server.get_server_info(), indent=2))

    print("\n--- Global Outbreak Data ---")
    result = server.call_tool("get_outbreak_data", {"region": "global"})
    print(json.dumps(json.loads(result["content"][0]["text"]), indent=2)[:2000])


if __name__ == "__main__":
    main()
