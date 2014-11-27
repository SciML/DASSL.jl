type Counter
    rejected :: Int             # rejected steps
    accepted :: Int             # accepted steps
    total :: Int                # total steps (should be equal to rejected+accepted)
    rejected_current :: Int     # number of rejected steps in a row
    fixed :: Int                # number of steps with fixed order and
                                # stepsize in a row
end

Counter() = Counter(0,0,0,0,0)

function accepted!(c::Counter)
    c.accepted += 1
    c.total += 1
    c.rejected_current = 0      # reset the counter
end

function rejected!(c::Counter)
    c.rejected += 1
    c.total += 1
    c.rejected_current += 1
end

function order_unchanged!(c::Counter)
    c.fixed += 1
end

function order_changed!(c::Counter)
    c.fixed = 1    # reset the counter
end
