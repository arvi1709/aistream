def mock_lead_capture(name: str, email: str, platform: str):
    """Controlled mock tool to capture a lead. Only to be called when all fields are present."""
    msg = f"Lead captured successfully: {name}, {email}, {platform}"
    print(msg)
    return {"status": "ok", "message": msg, "name": name, "email": email, "platform": platform}

