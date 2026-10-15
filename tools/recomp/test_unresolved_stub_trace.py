"""Regression checks for unresolved-call stub diagnostics."""

from tools.recomp.translator import _render_unresolved_stub


def test_unresolved_stub_records_target_before_restoring_guest_stack():
    generated = _render_unresolved_stub(
        0x00124F2C, "sub_00124F2C", popped=0)

    assert "RECOMP_UNRESOLVED_STUB_HIT(0x00124F2Cu);" in generated
    assert generated.index("RECOMP_UNRESOLVED_STUB_HIT") < generated.index(
        "g_esp += 4")


if __name__ == "__main__":
    test_unresolved_stub_records_target_before_restoring_guest_stack()
    print("1 checks passed")
