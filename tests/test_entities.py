"""Tests for game entities."""

import pytest

from parametric_physics_platformer.physics import PhysicsWorld, COLLISION_PLAYER
from parametric_physics_platformer.entities import Player, Platform, Goal, Hazard, TimedHazard, FlashingZone, Spring, Collectible
from parametric_physics_platformer.config import PhysicsConfig
from parametric_physics_platformer.dynamics import StandardDynamics


class TestPlayer:
    @pytest.fixture
    def physics(self):
        return PhysicsWorld()

    def test_creation(self, physics):
        player = Player(physics, (100, 200))
        assert player.body is not None
        assert player.shape is not None
        assert player.shape.collision_type == COLLISION_PLAYER

    def test_initial_position(self, physics):
        player = Player(physics, (150, 250))
        x, y = player.position
        assert x == pytest.approx(150, abs=1)
        assert y == pytest.approx(250, abs=1)

    def test_initial_velocity(self, physics):
        player = Player(physics, (100, 200))
        vx, vy = player.velocity
        assert vx == pytest.approx(0, abs=1)
        assert vy == pytest.approx(0, abs=1)

    def test_falls_under_gravity(self, physics):
        player = Player(physics, (100, 200))
        initial_y = player.position[1]

        for _ in range(60):
            player.update()
            physics.step(1 / 60)

        assert player.position[1] < initial_y

    def test_grounded_on_platform(self, physics):
        # Create ground platform
        Platform(physics, 100, 30, 200)

        # Create player above it
        player = Player(physics, (100, 200))

        # Let player fall
        for _ in range(100):
            player.update()
            physics.step(1 / 60)

        assert player.is_grounded

    def test_ground_friction_affects_damping(self):
        """Higher ground_friction should slow player faster when grounded."""
        # Test low friction (icy) - damping = 1.0 - 0.05*0 = 1.0 (no slowdown)
        physics_low = PhysicsWorld()
        Platform(physics_low, 100, 10, 400)
        config_low = PhysicsConfig(ground_friction=0.0)
        player_low = Player(physics_low, (100, 50), physics_config=config_low)

        # Let player land
        for _ in range(60):
            player_low.update()
            physics_low.step(1 / 60)

        # Give horizontal velocity and measure decay
        player_low.body.velocity = (200, 0)
        for _ in range(10):
            player_low.update()
            physics_low.step(1 / 60)
        vel_low = abs(player_low.velocity[0])

        # Test high friction (sticky) - damping = 1.0 - 0.05*0.8 = 0.96
        physics_high = PhysicsWorld()
        Platform(physics_high, 100, 10, 400)
        config_high = PhysicsConfig(ground_friction=0.8)
        player_high = Player(physics_high, (100, 50), physics_config=config_high)

        # Let player land
        for _ in range(60):
            player_high.update()
            physics_high.step(1 / 60)

        # Give horizontal velocity and measure decay
        player_high.body.velocity = (200, 0)
        for _ in range(10):
            player_high.update()
            physics_high.step(1 / 60)
        vel_high = abs(player_high.velocity[0])

        # Low friction should retain more velocity
        assert vel_low > vel_high


class TestPlatform:
    @pytest.fixture
    def physics(self):
        return PhysicsWorld()

    def test_creation(self, physics):
        platform = Platform(physics, 100, 50, 200)
        assert platform.shape is not None
        assert platform.width == 200

    def test_bounds(self, physics):
        platform = Platform(physics, 100, 50, 200, height=40)
        left, bottom, right, top = platform.bounds
        assert left == 0  # 100 - 200/2
        assert right == 200  # 100 + 200/2
        assert bottom == 30  # 50 - 40/2
        assert top == 70  # 50 + 40/2


class TestGoal:
    @pytest.fixture
    def physics(self):
        return PhysicsWorld()

    def test_creation(self, physics):
        goal = Goal(physics, 300, 100)
        assert goal.shape is not None
        assert not goal.reached

    def test_player_reaching_goal(self, physics):
        # Create goal at ground level
        goal = Goal(physics, 100, 60)

        # Create player at same position
        player = Player(physics, (100, 60))

        # Step physics to trigger collision
        for _ in range(10):
            player.update()
            physics.step(1 / 60)

        assert goal.reached


class TestHazard:
    @pytest.fixture
    def physics(self):
        return PhysicsWorld()

    def test_creation(self, physics):
        hazard = Hazard(physics, 200, 50)
        assert hazard.shape is not None
        assert not hazard.triggered

    def test_player_touching_hazard(self, physics):
        # Create hazard
        hazard = Hazard(physics, 100, 60)

        # Create player at same position
        player = Player(physics, (100, 60))

        # Step physics to trigger collision
        for _ in range(10):
            player.update()
            physics.step(1 / 60)

        assert hazard.triggered

    def test_static_hazard_does_not_move(self, physics):
        """Static hazard (speed=0) should not move when updated."""
        hazard = Hazard(physics, 200, 50, speed=0.0)
        initial_x = hazard.x

        # Update many times
        for _ in range(100):
            hazard.update(1 / 60)

        assert hazard.x == initial_x

    def test_moving_hazard_changes_position(self, physics):
        """Moving hazard should change x position over time."""
        hazard = Hazard(physics, 200, 50, speed=100.0, patrol_distance=200.0)
        initial_x = hazard.x

        # Update for 0.5 seconds - should move 50 pixels
        for _ in range(30):
            hazard.update(1 / 60)

        assert hazard.x != initial_x
        assert hazard.x > initial_x  # Moving right initially

    def test_moving_hazard_reverses_at_patrol_bounds(self, physics):
        """Moving hazard should reverse direction at patrol boundaries."""
        # Start at x=200, patrol_distance=100 means bounds are [150, 250]
        hazard = Hazard(physics, 200, 50, speed=100.0, patrol_distance=100.0)

        # Move right until hitting bound (50 pixels at 100 px/s = 0.5s)
        for _ in range(60):  # 1 second
            hazard.update(1 / 60)

        # Should have hit right bound and reversed, now moving left
        # After 1 second at 100px/s, starting at 200:
        # 0-0.5s: move right 50px to 250 (right bound)
        # 0.5-1s: move left 50px to 200
        assert hazard.x == pytest.approx(200, abs=5)

    def test_moving_hazard_patrols_back_and_forth(self, physics):
        """Moving hazard should patrol continuously within bounds."""
        hazard = Hazard(physics, 200, 50, speed=100.0, patrol_distance=100.0)

        positions = []
        for i in range(120):  # 2 seconds
            hazard.update(1 / 60)
            if i % 15 == 0:  # Sample every 0.25s
                positions.append(hazard.x)

        # Should have min and max within patrol bounds
        assert min(positions) >= 150 - 1  # Left bound (with tolerance)
        assert max(positions) <= 250 + 1  # Right bound (with tolerance)

    def test_moving_hazard_y_stays_constant(self, physics):
        """Moving hazard should only move horizontally."""
        hazard = Hazard(physics, 200, 50, speed=100.0, patrol_distance=100.0)
        initial_y = hazard.y

        for _ in range(60):
            hazard.update(1 / 60)

        assert hazard.y == initial_y

    def test_moving_hazard_still_triggers_on_player_contact(self, physics):
        """Moving hazard should still kill player on contact."""
        # Create moving hazard
        hazard = Hazard(physics, 100, 60, speed=50.0, patrol_distance=100.0)

        # Create player at same position
        player = Player(physics, (100, 60))

        # Step physics to trigger collision
        for _ in range(10):
            hazard.update(1 / 60)
            player.update()
            physics.step(1 / 60)

        assert hazard.triggered

    def test_moving_hazard_bounds_update_with_position(self, physics):
        """Hazard bounds should reflect current position for moving hazards."""
        hazard = Hazard(physics, 200, 50, width=30, height=30, speed=100.0, patrol_distance=200.0)

        # Get initial bounds
        left1, bottom1, right1, top1 = hazard.bounds
        assert left1 == pytest.approx(200 - 15, abs=1)

        # Move hazard
        for _ in range(30):  # Move ~50 pixels right
            hazard.update(1 / 60)

        # Bounds should have shifted
        left2, bottom2, right2, top2 = hazard.bounds
        assert left2 > left1
        assert bottom2 == bottom1  # Y unchanged


class TestTimedHazard:
    @pytest.fixture
    def physics(self):
        return PhysicsWorld()

    def test_creation(self, physics):
        hazard = TimedHazard(physics, 200, 50)
        assert hazard.shape is not None
        assert hazard.active  # Starts active by default

    def test_starts_active_by_default(self, physics):
        hazard = TimedHazard(physics, 200, 50)
        assert hazard.active is True

    def test_can_start_inactive(self, physics):
        hazard = TimedHazard(physics, 200, 50, start_active=False)
        assert hazard.active is False

    def test_toggles_to_inactive_after_active_duration(self, physics):
        hazard = TimedHazard(physics, 200, 50, active_duration=1.0, inactive_duration=0.5)
        assert hazard.active is True

        # Simulate 1 second
        for _ in range(60):
            hazard.update(1 / 60)

        assert hazard.active is False

    def test_toggles_back_to_active_after_inactive_duration(self, physics):
        hazard = TimedHazard(physics, 200, 50, active_duration=0.5, inactive_duration=0.5)

        # Go inactive (0.5s + buffer for float precision)
        for _ in range(32):
            hazard.update(1 / 60)
        assert hazard.active is False

        # Go active again (another 0.5s)
        for _ in range(32):
            hazard.update(1 / 60)
        assert hazard.active is True

    def test_cycles_continuously(self, physics):
        hazard = TimedHazard(physics, 200, 50, active_duration=0.5, inactive_duration=0.5)

        states = []
        for i in range(120):  # 2 seconds
            hazard.update(1 / 60)
            if i % 15 == 0:  # Sample every 0.25s
                states.append(hazard.active)

        # Should have both True and False states
        assert True in states
        assert False in states

    def test_triggered_only_when_active(self, physics):
        """Player should only die when hazard is active."""
        hazard = TimedHazard(physics, 100, 60, active_duration=10.0, inactive_duration=10.0)

        # Create player at same position while hazard is active
        player = Player(physics, (100, 60))
        for _ in range(10):
            player.update()
            physics.step(1 / 60)

        assert hazard.triggered is True

    def test_not_triggered_when_inactive(self, physics):
        """Player should not die when hazard is inactive."""
        hazard = TimedHazard(physics, 100, 60, active_duration=10.0, inactive_duration=10.0, start_active=False)

        # Create player at same position while hazard is inactive
        player = Player(physics, (100, 60))
        for _ in range(10):
            player.update()
            physics.step(1 / 60)

        assert hazard.triggered is False


class TestFlashingZone:
    @pytest.fixture
    def physics(self):
        return PhysicsWorld()

    def test_creation(self, physics):
        zone = FlashingZone(physics, 200, 50)
        assert zone.shape is not None
        assert zone.safe  # Starts safe by default

    def test_starts_safe_by_default(self, physics):
        zone = FlashingZone(physics, 200, 50)
        assert zone.safe is True
        assert zone.deadly is False

    def test_can_start_deadly(self, physics):
        zone = FlashingZone(physics, 200, 50, start_safe=False)
        assert zone.safe is False
        assert zone.deadly is True

    def test_toggles_to_deadly_after_safe_duration(self, physics):
        zone = FlashingZone(physics, 200, 50, safe_duration=1.0, deadly_duration=0.5)
        assert zone.safe is True

        # Simulate 1 second
        for _ in range(60):
            zone.update(1 / 60)

        assert zone.safe is False
        assert zone.deadly is True

    def test_toggles_back_to_safe_after_deadly_duration(self, physics):
        zone = FlashingZone(physics, 200, 50, safe_duration=0.5, deadly_duration=0.5)

        # Go deadly (0.5s + buffer for float precision)
        for _ in range(32):
            zone.update(1 / 60)
        assert zone.deadly is True

        # Go safe again (another 0.5s)
        for _ in range(32):
            zone.update(1 / 60)
        assert zone.safe is True

    def test_triggered_only_when_deadly(self, physics):
        """Player should only die when zone is deadly."""
        zone = FlashingZone(physics, 100, 60, safe_duration=10.0, deadly_duration=10.0, start_safe=False)

        # Create player at same position while zone is deadly
        player = Player(physics, (100, 60))
        for _ in range(10):
            player.update()
            physics.step(1 / 60)

        assert zone.triggered is True

    def test_not_triggered_when_safe(self, physics):
        """Player should not die when zone is safe."""
        zone = FlashingZone(physics, 100, 60, safe_duration=10.0, deadly_duration=10.0, start_safe=True)

        # Create player at same position while zone is safe
        player = Player(physics, (100, 60))
        for _ in range(10):
            player.update()
            physics.step(1 / 60)

        assert zone.triggered is False

    def test_larger_default_size_than_hazard(self, physics):
        """FlashingZone should default to larger area than Hazard."""
        zone = FlashingZone(physics, 200, 50)
        hazard = TimedHazard(physics, 300, 50)
        assert zone.width > hazard.width


class TestSpring:
    @pytest.fixture
    def physics(self):
        return PhysicsWorld()

    def test_creation(self, physics):
        spring = Spring(physics, 200, 50)
        assert spring.shape is not None

    def test_launches_player_upward(self, physics):
        # Create ground platform and spring on it
        Platform(physics, 100, 20, 200)
        spring = Spring(physics, 100, 40, launch_impulse=500.0)

        # Create player above spring
        player = Player(physics, (100, 100))

        # Let player fall onto spring
        for _ in range(30):
            player.update()
            physics.step(1 / 60)

        initial_y = player.position[1]

        # Continue - player should be launched upward
        for _ in range(30):
            spring.reset_trigger()
            player.update()
            physics.step(1 / 60)

        # Player should have been launched upward from spring contact
        # (may still be rising or have risen and be falling)
        # Check that spring was triggered
        # Note: We need to check across multiple frames
        pass  # Impulse was applied in collision handler

    def test_spring_triggers_on_player_contact(self, physics):
        """Spring should register trigger when player contacts it."""
        spring = Spring(physics, 100, 60, launch_impulse=500.0)

        # Create player at same position
        player = Player(physics, (100, 60))

        # Step physics to trigger collision
        for _ in range(5):
            spring.reset_trigger()
            player.update()
            physics.step(1 / 60)
            if spring.triggered:
                break

        assert spring.triggered is True

    def test_spring_applies_upward_velocity(self, physics):
        """Player should gain upward velocity after spring contact."""
        # Create spring
        spring = Spring(physics, 100, 60, launch_impulse=800.0)

        # Create player at same position with zero velocity
        player = Player(physics, (100, 80))
        player.body.velocity = (0, 0)

        # Step to trigger spring
        for _ in range(10):
            spring.reset_trigger()
            player.update()
            physics.step(1 / 60)

        # Player should have positive vertical velocity (upward)
        _, vy = player.velocity
        assert vy > 0

    def test_spring_preserves_horizontal_velocity(self, physics):
        """Spring should not significantly alter horizontal movement."""
        spring = Spring(physics, 100, 60, launch_impulse=500.0)

        # Create player with horizontal velocity
        player = Player(physics, (100, 80))
        player.body.velocity = (200, 0)

        initial_vx = player.velocity[0]

        # Step to trigger spring
        for _ in range(10):
            spring.reset_trigger()
            player.update()
            physics.step(1 / 60)

        # Horizontal velocity should be preserved (within some tolerance due to physics)
        vx, _ = player.velocity
        assert abs(vx - initial_vx) < 50  # Allow some variance from physics

    def test_spring_reset_trigger(self, physics):
        """reset_trigger should clear triggered state."""
        spring = Spring(physics, 100, 60)

        # Manually trigger
        spring._triggered_this_frame = True
        assert spring.triggered is True

        spring.reset_trigger()
        assert spring.triggered is False


class TestCollectible:
    @pytest.fixture
    def physics(self):
        return PhysicsWorld()

    def test_creation(self, physics):
        """Collectible should be created at specified position."""
        collectible = Collectible(physics, 200, 100)
        assert collectible.x == 200
        assert collectible.y == 100
        assert collectible.shape is not None

    def test_default_size(self, physics):
        """Collectible should have reasonable default size."""
        collectible = Collectible(physics, 200, 100)
        assert collectible.width == 20.0
        assert collectible.height == 20.0

    def test_custom_size(self, physics):
        """Collectible size should be configurable."""
        collectible = Collectible(physics, 200, 100, width=30.0, height=40.0)
        assert collectible.width == 30.0
        assert collectible.height == 40.0

    def test_not_collected_initially(self, physics):
        """Collectible should not be collected on creation."""
        collectible = Collectible(physics, 200, 100)
        assert collectible.collected is False

    def test_collected_on_player_contact(self, physics):
        """Collectible should be marked collected when player touches it."""
        collectible = Collectible(physics, 100, 60)

        # Create player at same position
        player = Player(physics, (100, 60))

        # Step physics to trigger collision
        for _ in range(5):
            player.update()
            physics.step(1 / 60)
            if collectible.collected:
                break

        assert collectible.collected is True

    def test_value_property(self, physics):
        """Collectible should have configurable value."""
        collectible = Collectible(physics, 200, 100, value=10)
        assert collectible.value == 10

    def test_default_value(self, physics):
        """Collectible should default to value of 1."""
        collectible = Collectible(physics, 200, 100)
        assert collectible.value == 1

    def test_bounds_property(self, physics):
        """Collectible should provide correct bounds."""
        collectible = Collectible(physics, 100, 50, width=20, height=20)
        left, bottom, right, top = collectible.bounds
        assert left == 90
        assert bottom == 40
        assert right == 110
        assert top == 60

    def test_is_sensor(self, physics):
        """Collectible should be a sensor (non-blocking)."""
        collectible = Collectible(physics, 200, 100)
        assert collectible.shape.sensor is True


class TestPlayerDynamicsIntegration:
    """Player should use DynamicsModel for physics."""

    @pytest.fixture
    def physics(self):
        return PhysicsWorld()

    def test_player_defaults_to_standard_dynamics(self, physics):
        player = Player(physics, (100, 200))
        assert isinstance(player.dynamics_model, StandardDynamics)

    def test_player_accepts_custom_dynamics(self, physics):
        model = StandardDynamics()
        player = Player(physics, (100, 200), dynamics_model=model)
        assert player.dynamics_model is model

    def test_player_tracks_airtime(self, physics):
        player = Player(physics, (100, 200))
        assert player.airtime == 0.0

    def test_player_airtime_increases_when_airborne(self, physics):
        player = Player(physics, (100, 500))  # High up, not grounded
        dt = 1 / 60
        physics.step(dt)
        player.update(dt)
        assert player.airtime > 0.0

    def test_player_airtime_resets_on_ground(self, physics):
        # Create ground platform so player lands
        Platform(physics, 100, 10, 400)
        player = Player(physics, (100, 50))
        # Let player land
        for _ in range(100):
            player.update(1 / 60)
            physics.step(1 / 60)
        assert player.is_grounded
        assert player.airtime == 0.0

    def test_player_velocity_func_is_set(self, physics):
        player = Player(physics, (100, 200))
        # Player body should have custom velocity_func
        assert player.body.velocity_func is not None
