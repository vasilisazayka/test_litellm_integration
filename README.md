# SAP ↔ LiteLLM Integration Test Automation

This repository contains automated GitHub Actions workflows that continuously validate the integration between **SAP AI Core** and **LiteLLM** using an extensive test script.

The goal is to **detect integration regressions early**, provide **transparent test history**, and enable **fast reaction** when something breaks.

---

## 🔄 Automated Test Workflows

The repository includes **two independent GitHub Actions workflows**:

### 1️⃣ Main Branch Testing (Daily)

- Runs the integration tests against **LiteLLM `main` branch**
- Ensures compatibility with the **latest ongoing development**
- Executed **daily**
- Helps detect breaking changes as early as possible

### 2️⃣ Release Version Testing (Twice per Week)

- Runs the same test suite against the **latest released LiteLLM version**
- Ensures stability of the **current production-ready release**
- Executed **twice per week**
- Validates real-world usage scenarios

---

## 📊 Test Reports & History

- Each workflow run generates:
  - A **detailed HTML test report**
  - A **historical statistics table** with pass/fail indicators
- All results are automatically published to **GitHub Pages**
- Reports are accumulated over time, providing **full test history and trends**

---

## 🚨 Automated Threshold Checks

At the end of each workflow run:

- Predefined **quality thresholds** are evaluated (e.g. error count, success rate)
- The workflow **fails immediately** if thresholds are exceeded
- This enables **quick alerting and investigation** when the integration degrades

---

## 🎯 Why This Exists

- Continuous validation of SAP ↔ LiteLLM integration
- Early detection of breaking changes
- Transparent and auditable test history
- Reduced risk when upgrading LiteLLM versions

---

📌 **All test results are publicly available via GitHub Pages for easy review and monitoring.**

