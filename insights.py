from collections import Counter

LOW_DETECTION_THRESHOLD = 5

def generate_insights(all_detections: list, filtered: list,
                      mode: str, parsed_query: dict,
                      confidence_label: str = "Low") -> str:
    counts = Counter(d["class"] for d in all_detections)
    obj = parsed_query.get("object", "objects")
    condition = parsed_query.get("condition", "all")
    total = len(all_detections)

    if total < LOW_DETECTION_THRESHOLD:
        return ("Low activity region detected — insufficient detections for reliable analysis. "
                "Try a higher-resolution image or a more active area.")

    matched = len(filtered)
    vehicles = sum(counts.get(c, 0) for c in ["car", "truck", "bus", "motorcycle"])
    people = counts.get("person", 0)

    lines = [
        f"📊 **Total detections:** {total}",
        f"🎯 **Matched '{obj}' ({condition}):** {matched}",
        f"🔬 Query confidence: {confidence_label}",
        "",
    ]

    if matched == 0:
        lines.append("No matching results for this query.")
        return "\n".join(lines)

    # Multi-signal smart rules
    if vehicles > 15 and people < 3:
        lines.append("High vehicles + low people → likely parking or idle zone.")
    if vehicles > 10 and people > 10:
        lines.append("High vehicles + high people → active urban zone.")
    if people > 15 and vehicles < 5:
        lines.append("High pedestrian activity + low vehicles → pedestrian zone or event area.")
    if lines[-1] != "":
        lines.append("")

    if mode == "Urban Planning":
        lines.append("🏙️ **Urban Planning Analysis**")
        lines.append(f"- Vehicles detected: {vehicles}")
        if vehicles > 30:
            lines.append("⚠️ Very high vehicle density — likely dense urban zone.")
        elif vehicles > 10:
            lines.append("🟡 Moderate vehicle density — suburban characteristics.")
        else:
            lines.append("🟢 Low density — rural or low-traffic area.")
        lines.append("ℹ️ Note: YOLO detects vehicles, not buildings. High vehicle density implies urban infrastructure.")

    elif mode == "Disaster Monitoring":
        lines.append("🚨 **Disaster Monitoring Analysis**")
        lines.append(f"- People detected: {people}")
        lines.append(f"- Vehicles in zone: {vehicles}")
        if people > 10:
            lines.append("⚠️ High pedestrian presence — possible evacuation or gathering area.")
        if vehicles > 15:
            lines.append("🚗 High vehicle count — possible emergency response or evacuation convoy.")
        if total == 0:
            lines.append("⬛ No detections — area may be deserted or obscured.")

    elif mode == "Traffic Analysis":
        lines.append("🚦 **Traffic Analysis**")
        cars = counts.get("car", 0)
        trucks = counts.get("truck", 0)
        buses = counts.get("bus", 0)
        lines.append(f"- Cars: {cars} | Trucks: {trucks} | Buses: {buses}")
        density_label = "🔴 High" if vehicles > 20 else "🟡 Moderate" if vehicles > 8 else "🟢 Low"
        lines.append(f"- Traffic activity density: **{density_label}**")
        if trucks > cars:
            lines.append("🏭 Heavy vehicle dominance — possible industrial or freight route.")

    return "\n".join(lines)
