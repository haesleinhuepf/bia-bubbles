import pygame
import numpy as np
from math import cos, sin, pi, atan2, dist
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import pyclesperanto as cle

class ImageScatter:
    """A class that represents an image on the canvas with both numpy array and pygame surface representations."""
    
    def __init__(self, array, pos=None, parent_id=None, function_name=None, image_type=None, name=None):
        """Initialize an ImageScatter with a numpy array.
        
        Args:
            array: Numpy array representation of the image
            pos: Position to place the image at (if None, will be centered)
            parent_id: ID of the parent image (if this is a processed result)
            function_name: Name of the function used to process the image
            image_type: Type of the image ('intensity', 'binary', or 'label')
            name: Display name of the image
        """
        self.array = array
        self.pos = pos
        self.parent_id = parent_id
        self.function_name = function_name
        self.image_type = image_type
        self.name = name
        
        # Add velocity and momentum tracking
        self.velocity = [0.0, 0.0]  # Current velocity [x, y]
        self.target_pos = pos  # Position this image is trying to reach
        self.follow_delay = 0.1  # How quickly this image follows its target (0-1)
        self.momentum = 0.95  # How much velocity is preserved each frame (0-1)
        
        # Add animation state tracking
        self.animation_start_time = None  # When the animation started
        self.animation_duration = 0.5  # Duration of the animation in seconds
        self.animation_scale = 0.0  # Current scale of the animation (0 to 1)
        self.is_animating = False  # Whether this image is currently animating
        
        # Convert numpy array to pygame surface for display
        if image_type == 'label':
            # For label images, use a custom colormap
            display_array = self._label_to_rgb(array)
        else:
            # For intensity and binary images, use the standard conversion
            if array.max() <= 1.0:
                display_array = array * 255
            else:
                display_array = array
                
            # Ensure the array is 2D (grayscale) or 3D (RGB)
            if len(display_array.shape) == 2:
                # For grayscale images, we need to create a 3D array with the same value in all channels
                display_array = np.stack([display_array] * 3, axis=-1)
            
        # Convert to uint8 for pygame
        display_array = display_array.astype(np.uint8)
        print(f"Final display array shape: {display_array.shape}")
        print(f"Final display array min: {display_array.min()}, max: {display_array.max()}")
            
        self.surface = pygame.surfarray.make_surface(display_array)
        
    def _label_to_rgb(self, array):
        """Convert a label image to RGB using a custom colormap.
        
        Args:
            array: Label image array with integer values
            
        Returns:
            RGB array with shape (height, width, 3)
        """
        # Get the maximum label value to determine the size of our color map
        max_label = np.max(array)
        
        # Create a color map for all possible labels (0 to max_label)
        # We'll use a deterministic approach to generate colors for labels beyond our predefined ones
        color_map = np.zeros((max_label + 1, 3), dtype=np.uint8)
        
        # Define colors for the first 10 labels
        predefined_colors = [
            [0, 0, 0],          # 0: Black for background
            [173, 216, 230],    # 1: Light blue
            [255, 165, 0],      # 2: Orange
            [50, 205, 50],      # 3: Lime green
            [255, 0, 0],        # 4: Red
            [128, 0, 128],      # 5: Purple
            [255, 255, 0],      # 6: Yellow
            [0, 255, 255],      # 7: Cyan
            [255, 192, 203],    # 8: Pink
            [165, 42, 42],      # 9: Brown
        ]
        
        # Assign predefined colors
        for i, color in enumerate(predefined_colors):
            if i <= max_label:
                color_map[i] = color
        
        # For labels beyond our predefined colors, generate deterministic colors
        # using a hash function approach
        for i in range(10, max_label + 1):
            # Use a simple hash function to generate RGB values
            # This ensures the same label always gets the same color
            r = (i * 13) % 256
            g = (i * 17) % 256
            b = (i * 19) % 256
            color_map[i] = [r, g, b]
        
        # Create the RGB image by indexing into the color map
        # This is a single vectorized operation
        rgb = color_map[array]
        
        return rgb
        
    def get_surface(self):
        """Get the pygame surface representation."""
        return self.surface
        
    def get_array(self):
        """Get the numpy array representation."""
        return self.array
        
    def get_pos(self):
        """Get the position of the image."""
        return self.pos
        
    def set_pos(self, pos):
        """Set the position of the image."""
        self.pos = pos
        
    def get_parent_id(self):
        """Get the parent ID of the image."""
        return self.parent_id
        
    def get_function_name(self):
        """Get the function name used to process the image."""
        return self.function_name
        
    def get_image_type(self):
        """Get the type of the image."""
        return self.image_type
        
    def set_image_type(self, image_type):
        """Set the type of the image."""
        self.image_type = image_type

    def get_name(self):
        """Get the display name of the image."""
        return self.name
        
    def set_name(self, name):
        """Set the display name of the image."""
        self.name = name

    def set_velocity(self, vx, vy):
        """Set the current velocity of the image."""
        self.velocity = [vx, vy]
        
    def get_velocity(self):
        """Get the current velocity of the image."""
        return self.velocity
        
    def set_target_pos(self, pos):
        """Set the target position this image is trying to reach."""
        self.target_pos = pos
        
    def get_target_pos(self):
        """Get the target position this image is trying to reach."""
        return self.target_pos
        
    def set_follow_delay(self, delay):
        """Set how quickly this image follows its target (0-1)."""
        self.follow_delay = max(0.01, min(1.0, delay))
        
    def get_follow_delay(self):
        """Get how quickly this image follows its target (0-1)."""
        return self.follow_delay


class ImageProcessingCanvas:
    def __init__(self, width=800, height=600, function_collection=None):
        pygame.init()
        # Set window title
        pygame.display.set_caption("Fun with Images")
        # Create resizable window
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        
        self.images = {}  # {id: ImageScatter}
        self.connections = []
        self.temporary_proposals = {}  # {id: ImageScatter}
        
        # Add proposal animation tracking
        self.proposal_queue = []  # Queue of proposals to animate
        self.last_proposal_time = 0  # Time of last proposal animation
        self.proposal_delay = 0.1  # Delay between proposals in seconds
        self.proposal_start_time = None  # When the proposal sequence started
        self.is_animating_proposals = False  # Whether we're currently showing proposals
        
        self.width = width
        self.height = height
        self.scale = 1.0
        self.rotation = 0
        
        self.dragging = False
        self.dragged_image_id = None  # Track which image is being dragged
        self.drag_offset = (0, 0)  # Offset from mouse to image center when dragging
        self.multi_touch = False
        self.touch_points = []
        self.initial_touch_distance = 0
        self.initial_touch_angle = 0
        
        # Add panning variables
        self.panning = False
        self.pan_start = None
        self.view_offset_x = 0
        self.view_offset_y = 0
        
        # Add relaxation variables
        self.relaxing = False
        self.relaxation_speed = 0.05  # Reduced from 0.1 to make movement slower
        self.min_distance = 150  # Minimum distance between images
        self.relaxation_strength = 0.3  # Reduced from 0.5 to make repulsion softer
        self.position_preference = 0.9  # Increased from 0.8 to make images stay more in place
        self.related_position_preference = 0.8  # Increased from 0.6 to make related images stay closer
        
        # Add click detection variables
        self.click_start_time = 0
        self.click_start_pos = None
        self.clicked_image_id = None
        self.click_threshold = 200  # milliseconds to distinguish click from drag
        
        self.function_collection = function_collection or {}
        
    def add_image(self, array, parent_id=None, pos=None, image_type=None):
        """Add a new image to the canvas.
        
        Args:
            array: Numpy array representation of the image
            parent_id: ID of the parent image (if this is a processed result)
            pos: Position to place the image at (if None, will be centered)
            image_type: Type of the image ('intensity', 'binary', or 'label'). If None, will be determined automatically.
        """
        img_id = len(self.images)
        if pos is None:
            pos = (self.width/2, self.height/2)
            
        # Determine the image type if not provided
        if image_type is None:
            if parent_id is not None and parent_id in self.images:
                # For processed images, determine type based on the array
                image_type = self.determine_image_type_from_array(array)
            else:
                # For initial images, assume intensity
                image_type = 'intensity'
        
        # Set the name based on whether this is the first image or a processed image
        name = None
        if img_id == 0:
            name = "original"
        elif parent_id is not None and parent_id in self.images:
            # For processed images, use the function name
            parent_img = self.images[parent_id]
            if parent_img.get_function_name():
                name = parent_img.get_function_name()
            
        image_scatter = ImageScatter(array, pos, parent_id, image_type=image_type, name=name)
        self.images[img_id] = image_scatter
        return img_id
        
    def determine_image_type_from_array(self, array):
        """Determine the type of image from its array representation.
        
        Args:
            array: Numpy array representation of the image
            
        Returns:
            str: 'intensity', 'binary', or 'label'
        """
        # Check if it's a binary image (only 0 and 1 values)
        unique_values = np.unique(array)
        
        if len(unique_values) <= 2 and (0 in unique_values or 1 in unique_values):
            return 'binary'
        
        # Check if it's a labeled image (integer values with gaps)
        if np.issubdtype(array.dtype, np.integer):
            # Labeled images typically have integer values with gaps between them
            # and a relatively small number of unique values compared to the range
            value_range = np.max(array) - np.min(array)
            if value_range > 0 and len(unique_values) < value_range / 2:
                return 'label'
        
        return 'intensity'
        
    def get_image_type(self, img_id):
        """Get the type of an image from the ImageScatter object.
        
        Args:
            img_id: ID of the image to check
            
        Returns:
            str: 'intensity', 'binary', or 'label'
        """
        if img_id in self.images:
            return self.images[img_id].get_image_type()
        return 'intensity'  # Default
        
    def process_image(self, img_id):
        # Clear old proposals
        self.temporary_proposals.clear()
        self.proposal_queue.clear()
        self.proposal_start_time = pygame.time.get_ticks() / 1000.0  # Convert to seconds
        self.is_animating_proposals = True  # Start animation sequence
        
        # Determine the image type
        img_type = self.get_image_type(img_id)
        
        # Update the image type in the ImageScatter object
        self.images[img_id].set_image_type(img_type)
        
        # Get the appropriate functions for this image type
        if img_type in self.function_collection:
            type_functions = self.function_collection[img_type]
            
            # Calculate total number of proposals
            total_functions = sum(len(funcs) for funcs in type_functions.values())
            
            # Determine initial distance based on number of proposals
            # More proposals need more space
            base_distance = 150
            if total_functions > 5:
                base_distance = 200
            if total_functions > 10:
                base_distance = 250
            if total_functions > 15:
                base_distance = 300
            print("base_distance", base_distance)
                
            # Scale the base distance by the current zoom level to maintain consistent visual distance
            base_distance = base_distance * self.scale
                
            # Keep track of the current function index across all categories
            current_function_index = 0
            
            # Generate proposals for each category in the type functions
            for category, functions in type_functions.items():
                for name, func in functions.items():
                    try:
                        # Process image using numpy array
                        result_array = func(self.images[img_id].get_array())
                        
                        # Determine the result image type
                        result_type = self.determine_result_type(img_type, category, name)
                        
                        # Calculate position avoiding overlap
                        base_pos = self.images[img_id].get_pos()
                        distance = base_distance
                        
                        # Calculate the angle for this proposal
                        # We want to distribute proposals in a fan-like pattern
                        # from top (-pi/2) to bottom (pi/2)
                        angle_step = 2 * pi / (total_functions) if total_functions > 1 else pi/4
                        angle = -pi/2 + current_function_index * angle_step
                        
                        # Position all proposals at the same distance but in a fan pattern
                        # This ensures all proposals have exactly the same distance from the center
                        pos = (
                            base_pos[0] + cos(angle) * distance,
                            base_pos[1] + sin(angle) * distance
                        )
                        
                        # Create a new ImageScatter for the proposal
                        proposal = ImageScatter(result_array, pos, parent_id=img_id, 
                                              function_name=f"{category}/{name}", 
                                              image_type=result_type,
                                              name=f"{category}/{name}")
                        
                        # Add to queue instead of directly to temporary_proposals
                        self.proposal_queue.append(proposal)
                        
                        # Increment the function index
                        current_function_index += 1
                    except Exception as e:
                        print(f"Error processing image with {category}/{name}: {e}")
    
    def determine_result_type(self, input_type, category, function_name):
        """Determine the type of the result image based on the input type and function.
        
        Args:
            input_type: Type of the input image ('intensity', 'binary', or 'label')
            category: Category of the function
            function_name: Name of the function
            
        Returns:
            str: 'intensity', 'binary', or 'label'
        """
        # Threshold functions convert intensity to binary
        if input_type == 'intensity' and category == 'threshold':
            return 'binary'
            
        # Connected components convert binary to label
        if input_type == 'binary' and function_name == 'connected_components':
            return 'label'
            
        # Measurement functions convert label to intensity
        if input_type == 'label' and category == 'measure':
            return 'intensity'
            
        # Most other functions preserve the input type
        return input_type
    
    def position_overlaps(self, pos, margin=100):
        # Scale the margin with the current zoom level
        scaled_margin = margin * self.scale
        # Check overlap with existing images
        for img in self.images.values():
            if dist(pos, img.get_pos()) < scaled_margin:
                return True
        # Check overlap with other proposals
        for prop in self.temporary_proposals.values():
            if dist(pos, prop.get_pos()) < scaled_margin:
                return True
        return False
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            # Handle window resize
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.size
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
            
            # Handle ESC key to clear proposals
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # Clear proposals if any exist
                    if self.temporary_proposals:
                        self.temporary_proposals.clear()
                        # Start relaxation after clearing proposals
                        self.start_relaxation()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Convert screen position to world position
                    world_pos = self.screen_to_world(event.pos)
                    
                    # Check temporary proposals first
                    closest_proposal_id = None
                    closest_distance = float('inf')
                    
                    for prop_id, prop in self.temporary_proposals.items():
                        if self.point_in_image(world_pos, prop_id, is_proposal=True):
                            # Calculate distance to this proposal
                            distance = self.distance_to_image(world_pos, prop_id, is_proposal=True)
                            if distance < closest_distance:
                                closest_distance = distance
                                closest_proposal_id = prop_id
                    
                    # If we found a proposal, select the closest one
                    if closest_proposal_id is not None:
                        # Get the proposal's image type before adding it
                        proposal_type = self.temporary_proposals[closest_proposal_id].get_image_type()
                        # Get the proposal's function name
                        proposal_function_name = self.temporary_proposals[closest_proposal_id].get_function_name()
                        # Get the parent_id from the proposal
                        parent_id = self.temporary_proposals[closest_proposal_id].get_parent_id()
                        # Add the proposal as a new image
                        new_img_id = self.add_image(self.temporary_proposals[closest_proposal_id].get_array(), 
                                     parent_id=parent_id,  # Use the proposal's parent_id
                                     pos=self.temporary_proposals[closest_proposal_id].get_pos(),
                                     image_type=proposal_type)
                        # Set the name of the new image to the function name
                        if proposal_function_name:
                            self.images[new_img_id].set_name(proposal_function_name)
                        self.temporary_proposals.clear()
                        # Start relaxation after adding a new image
                        self.start_relaxation()
                    # Then check permanent images
                    else:
                        image_clicked = False
                        for img_id in self.images:
                            if self.point_in_image(world_pos, img_id):
                                # Store click information for potential drag or click
                                self.click_start_time = pygame.time.get_ticks()
                                self.click_start_pos = world_pos
                                self.clicked_image_id = img_id
                                # Calculate offset from mouse to image center for potential drag
                                img_pos = self.images[img_id].get_pos()
                                self.drag_offset = (img_pos[0] - world_pos[0], img_pos[1] - world_pos[1])
                                image_clicked = True
                                break
                        
                        # If no image was clicked and we have proposals, clear them
                        if not image_clicked and self.temporary_proposals:
                            self.temporary_proposals.clear()
                            # Start relaxation after clearing proposals
                            self.start_relaxation()
                            # Allow panning with left mouse button
                            self.panning = True
                            self.pan_start = event.pos
                        # If no image was clicked and no proposals, allow panning
                        elif not image_clicked:
                            self.panning = True
                            self.pan_start = event.pos
                
                elif event.button == 3:  # Right click
                    # Check if we clicked on an image
                    world_pos = self.screen_to_world(event.pos)
                    image_clicked = False
                    for img_id in self.images:
                        if self.point_in_image(world_pos, img_id):
                            image_clicked = True
                            break
                    
                    if image_clicked:
                        # If we clicked on an image, handle rotation
                        self.dragging = True
                        self.drag_start = event.pos
                        self.initial_rotation = self.rotation
                    else:
                        # If we didn't click on an image, handle panning
                        self.panning = True
                        self.pan_start = event.pos
                
                elif event.button == 2:  # Middle click
                    self.panning = True
                    self.pan_start = event.pos
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left click
                    # Check if this was a click (short press) or a drag
                    if self.clicked_image_id is not None and self.clicked_image_id in self.images:
                        current_time = pygame.time.get_ticks()
                        time_diff = current_time - self.click_start_time
                        
                        # If it was a short press and the mouse hasn't moved much, treat as a click
                        if time_diff < self.click_threshold and not self.dragging:
                            # Process the image to show proposals
                            self.process_image(self.clicked_image_id)
                    
                    # Reset all states
                    self.dragging = False
                    self.dragged_image_id = None
                    self.clicked_image_id = None
                    self.click_start_pos = None
                    self.panning = False
                    
                    # Start relaxation if we were dragging
                    if self.dragging:
                        self.start_relaxation()
                elif event.button == 3:  # Right click
                    self.dragging = False
                    self.panning = False
                elif event.button == 2:  # Middle click
                    self.panning = False
            
            elif event.type == pygame.MOUSEMOTION:
                # Check if left mouse button is pressed
                if pygame.mouse.get_pressed()[0]:  # Left button
                    # Check if we should start dragging
                    if self.clicked_image_id is not None and not self.dragging:
                        # Convert screen position to world position
                        world_pos = self.screen_to_world(event.pos)
                        
                        # Check if the mouse has moved enough to consider it a drag
                        if self.click_start_pos is not None:
                            dx = world_pos[0] - self.click_start_pos[0]
                            dy = world_pos[1] - self.click_start_pos[1]
                            distance = (dx*dx + dy*dy) ** 0.5
                            
                            # If moved more than 5 pixels, start dragging
                            if distance > 5:
                                self.dragging = True
                                self.dragged_image_id = self.clicked_image_id
                    
                    # Handle dragging if active
                    if self.dragging and self.dragged_image_id is not None:
                        # Handle dragging an image
                        world_pos = self.screen_to_world(event.pos)
                        # Calculate new position for the dragged image
                        new_pos = (
                            world_pos[0] + self.drag_offset[0],
                            world_pos[1] + self.drag_offset[1]
                        )
                        # Try to move the image and its related images
                        move_success = self.move_image_group(self.dragged_image_id, new_pos)
                        
                        # If the move failed (image was removed), reset dragging state
                        if not move_success:
                            self.dragging = False
                            self.dragged_image_id = None
                            self.clicked_image_id = None
                            self.click_start_pos = None
                    elif self.panning and self.clicked_image_id is None:  # Only pan if we're not clicking on an image
                        dx = event.pos[0] - self.pan_start[0]
                        dy = event.pos[1] - self.pan_start[1]
                        self.view_offset_x += dx
                        self.view_offset_y += dy
                        self.pan_start = event.pos
                # Check if middle mouse button is pressed
                elif pygame.mouse.get_pressed()[1]:  # Middle button
                    if not self.panning:
                        self.panning = True
                        self.pan_start = event.pos
                    else:
                        dx = event.pos[0] - self.pan_start[0]
                        dy = event.pos[1] - self.pan_start[1]
                        self.view_offset_x += dx
                        self.view_offset_y += dy
                        self.pan_start = event.pos
                # Check if right mouse button is pressed
                elif pygame.mouse.get_pressed()[2]:  # Right button
                    if self.dragging:
                        # Handle rotation
                        dx = event.pos[0] - self.drag_start[0]
                        self.rotation = self.initial_rotation + dx * 0.01
                    elif self.panning:
                        # Handle panning
                        dx = event.pos[0] - self.pan_start[0]
                        dy = event.pos[1] - self.pan_start[1]
                        self.view_offset_x += dx
                        self.view_offset_y += dy
                        self.pan_start = event.pos
            
            elif event.type == pygame.MOUSEWHEEL:
                # Get mouse position before zoom
                mouse_pos = pygame.mouse.get_pos()
                
                # Calculate zoom factor
                zoom_factor = 1.1 if event.y > 0 else 0.9
                
                # Apply zoom relative to mouse position
                self.zoom_at_point(mouse_pos, zoom_factor)
            
            # Touch events
            elif event.type == pygame.FINGERDOWN:
                touch_pos = (event.x * self.screen.get_width(),
                           event.y * self.screen.get_height())
                self.touch_points.append(touch_pos)
                if len(self.touch_points) == 2:
                    # Two fingers down - enable multi-touch and disable panning
                    self.multi_touch = True
                    self.panning = False  # Disable panning when multi-touch is active
                    self.initial_touch_distance = dist(*self.touch_points)
                    self.initial_touch_angle = atan2(self.touch_points[1][1] - self.touch_points[0][1],
                                                   self.touch_points[1][0] - self.touch_points[0][0])
                else:
                    # Single touch for panning
                    self.panning = True
                    self.multi_touch = False  # Disable multi-touch when panning is active
                    self.pan_start = touch_pos
            
            elif event.type == pygame.FINGERUP:
                # Remove the finger that was lifted
                finger_id = event.finger_id
                if finger_id < len(self.touch_points):
                    self.touch_points.pop(finger_id)
                
                # Update touch states based on remaining fingers
                if len(self.touch_points) == 0:
                    # No fingers left - disable both panning and multi-touch
                    self.panning = False
                    self.multi_touch = False
                elif len(self.touch_points) == 1:
                    # One finger left - enable panning and disable multi-touch
                    self.panning = True
                    self.multi_touch = False
                    # Update pan start position to the remaining finger
                    self.pan_start = self.touch_points[0]
                # If two fingers remain, multi-touch stays enabled
            
            elif event.type == pygame.FINGERMOTION:
                # Update the touch points list with the current position
                finger_id = event.finger_id
                if finger_id < len(self.touch_points):
                    self.touch_points[finger_id] = (event.x * self.screen.get_width(),
                                                  event.y * self.screen.get_height())
                
                # Check if we have multiple fingers touching the screen
                if len(self.touch_points) >= 2:
                    # We have multiple fingers - handle multi-touch for zoom and rotation
                    self.multi_touch = True
                    self.panning = False  # Disable panning when multi-touch is active
                    
                    # Use only the first two touch points for calculations
                    # This ensures we don't get more than 2 arguments for dist()
                    current_points = self.touch_points[:2]
                    current_distance = dist(*current_points)
                    
                    # Calculate zoom factor based on the ratio of current distance to initial distance
                    # This makes zooming proportional to finger movement
                    if self.initial_touch_distance > 0:
                        # Calculate the ratio of current distance to initial distance
                        distance_ratio = current_distance / self.initial_touch_distance
                        
                        # Apply a scaling factor to make the zoom more manageable
                        # This dampens the effect of the ratio to prevent too rapid zooming
                        scaling_factor = 0.1  # Adjust this value to control zoom sensitivity
                        
                        # Calculate the zoom factor based on the ratio and scaling
                        # If ratio > 1, we're zooming in; if ratio < 1, we're zooming out
                        zoom_factor = 1.0 + (distance_ratio - 1.0) * scaling_factor
                        
                        # Ensure the zoom factor stays within reasonable bounds
                        zoom_factor = max(0.95, min(1.05, zoom_factor))
                        
                        # Calculate center point for zoom (midpoint between the two fingers)
                        center = ((current_points[0][0] + current_points[1][0])/2,
                                 (current_points[0][1] + current_points[1][1])/2)
                                 
                        # Apply zoom relative to center point
                        self.zoom_at_point(center, zoom_factor)
                    
                    # Handle rotation
                    current_angle = atan2(current_points[1][1] - current_points[0][1],
                                        current_points[1][0] - current_points[0][0])
                    self.rotation += current_angle - self.initial_touch_angle
                    
                    # Update initial values for next frame
                    self.initial_touch_distance = current_distance
                    self.initial_touch_angle = current_angle
                elif len(self.touch_points) == 1:
                    # Single finger - handle panning
                    self.multi_touch = False
                    self.panning = True
                    
                    # Handle single-touch for panning
                    current_pos = (event.x * self.screen.get_width(),
                                 event.y * self.screen.get_height())
                    dx = current_pos[0] - self.pan_start[0]
                    dy = current_pos[1] - self.pan_start[1]
                    self.view_offset_x += dx
                    self.view_offset_y += dy
                    self.pan_start = current_pos
        
        return True

    def zoom_at_point(self, point, zoom_factor):
        """Zoom in/out relative to a specific point on the screen.
        
        Args:
            point: The (x, y) point to zoom relative to
            zoom_factor: The factor to zoom by (1.1 for zoom in, 0.9 for zoom out)
        """
        # Store the old scale
        old_scale = self.scale
        
        # Apply the new scale
        self.scale *= zoom_factor
        
        # Create a list of all items to update (both images and proposals)
        all_items = []
        
        # Add permanent images
        for img in self.images.values():
            all_items.append(img)
            
        # Add temporary proposals
        for prop in self.temporary_proposals.values():
            all_items.append(prop)
            
        # Update positions for all items
        for item in all_items:
            pos = item.get_pos()
            # Calculate vector from mouse to image
            dx = pos[0] - point[0]
            dy = pos[1] - point[1]
            # Scale this vector by the zoom factor
            new_dx = dx * zoom_factor
            new_dy = dy * zoom_factor
            # Set new position
            item.set_pos((point[0] + new_dx, point[1] + new_dy))

    def run(self):
        """Main game loop"""
        running = True
        while running:
            running = self.handle_events()
            self.update_relaxation()  # Add relaxation update
            self.render()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()
        
    def render(self):
        """Render all images and proposals"""
        self.screen.fill((0, 0, 0))  # Black background
        
        # Draw connections between related images
        self.render_connections()
        
        # Create a list of all items to render (both images and proposals)
        all_items = []
        
        # Add permanent images
        for img_id, img in self.images.items():
            all_items.append((img.get_surface(), img.get_pos(), None, img.get_name()))
            
        # Add temporary proposals
        for prop in self.temporary_proposals.values():
            all_items.append((prop.get_surface(), prop.get_pos(), prop.get_function_name(), None))
            
        # Render all items using the same method
        for surface, pos, function_name, image_name in all_items:
            self.render_image(surface, pos, function_name, image_name)
        
    def render_image(self, surface, pos, function_name=None, image_name=None):
        """Render a single image at the given position with a circular mask"""
        # Get the current proposal if this is a proposal
        current_proposal = None
        if function_name:  # This indicates it's a proposal
            for prop in self.temporary_proposals.values():
                if prop.get_function_name() == function_name:
                    current_proposal = prop
                    break
        
        # Apply transformations
        transformed_surface = pygame.transform.rotozoom(surface, self.rotation, self.scale)
        
        # Apply view offset to position
        adjusted_pos = (pos[0] + self.view_offset_x, pos[1] + self.view_offset_y)
        rect = transformed_surface.get_rect(center=adjusted_pos)
        
        # Create a circular mask
        mask_surface = pygame.Surface(transformed_surface.get_size(), pygame.SRCALPHA)
        radius = min(transformed_surface.get_width(), transformed_surface.get_height()) * 0.4
        
        # If this is a proposal and it's animating, scale the radius
        if current_proposal and current_proposal.is_animating:
            radius *= current_proposal.animation_scale
        
        center = (transformed_surface.get_width() // 2, transformed_surface.get_height() // 2)
        pygame.draw.circle(mask_surface, (255, 255, 255, 255), center, radius)
        
        # Create a temporary surface for the masked image
        temp_surface = pygame.Surface(transformed_surface.get_size(), pygame.SRCALPHA)
        temp_surface.blit(transformed_surface, (0, 0))
        
        # Apply the mask
        temp_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        
        # Render the masked image
        self.screen.blit(temp_surface, rect)
        
        # Draw white outline around the circle
        pygame.draw.circle(self.screen, (255, 255, 255), adjusted_pos, radius, 2)
        
        # Render name if provided (for both proposals and images)
        if function_name:
            self.render_name(pos, function_name, radius, is_proposal=True)
        elif image_name:
            self.render_name(pos, image_name, radius, is_proposal=False)
    
    def render_name(self, pos, name, radius, is_proposal=False):
        """Render a name (either function name for proposals or image name) with appropriate styling.
        
        Args:
            pos: Position of the image
            name: Name to render
            radius: Radius of the image circle
            is_proposal: Whether this is a proposal name (True) or an image name (False)
        """
        if name is None:
            return
            
        # Scale the radius with the current scale to maintain proper distance
        scaled_radius = radius * self.scale
        
        # Scale font size with the current scale
        font_size = int(24 * self.scale)
        font = pygame.font.Font(None, font_size)
        
        # Extract just the name part after the "/"
        display_name = name.split('/')[-1] if '/' in name else name
        
    
        # For images, position text at the center with white outline
        # Position text at the center of the image
        text_x = pos[0]
        text_y = pos[1]
        
        # Apply view offset to text position
        text_x += self.view_offset_x
        text_y += self.view_offset_y
        
        # Split by "_" for multi-line display
        lines = display_name.split('_')
        
        # Calculate total height of all lines
        line_height = font.get_linesize()
        total_height = line_height * len(lines)
        
        # Calculate starting y position to center all lines vertically
        start_y = text_y - (total_height - line_height) / 2
        
        # Render each line
        for i, line in enumerate(lines):
            # Create text surfaces for this line
            text_white = font.render(line, True, (255, 255, 255))
            text_black = font.render(line, True, (0, 0, 0))
            
            # Calculate y position for this line
            line_y = start_y + i * line_height
            
            # Get the rects for all text surfaces
            text_rect_white_tl = text_white.get_rect(center=(text_x - 1, line_y - 1))  # Top-left
            text_rect_white_tr = text_white.get_rect(center=(text_x + 1, line_y - 1))  # Top-right
            text_rect_white_bl = text_white.get_rect(center=(text_x - 1, line_y + 1))  # Bottom-left
            text_rect_white_br = text_white.get_rect(center=(text_x + 1, line_y + 1))  # Bottom-right
            text_rect_black = text_black.get_rect(center=(text_x, line_y))  # Center
            
            # Render all text surfaces for this line
            self.screen.blit(text_white, text_rect_white_tl)
            self.screen.blit(text_white, text_rect_white_tr)
            self.screen.blit(text_white, text_rect_white_bl)
            self.screen.blit(text_white, text_rect_white_br)
            self.screen.blit(text_black, text_rect_black)
    
    def screen_to_world(self, screen_pos):
        """Convert screen coordinates to world coordinates"""
        # Subtract the view offset to get world coordinates
        return (screen_pos[0] - self.view_offset_x, screen_pos[1] - self.view_offset_y)
        
    def point_in_image(self, point, img_id, is_proposal=False):
        """Check if a point is within an image"""
        # Get the appropriate surface and position based on whether it's a proposal or not
        if is_proposal:
            surface = self.temporary_proposals[img_id].get_surface()
            pos = self.temporary_proposals[img_id].get_pos()
        else:
            surface = self.images[img_id].get_surface()
            pos = self.images[img_id].get_pos()
            
        # Apply view offset to position
        adjusted_pos = (pos[0] + self.view_offset_x, pos[1] + self.view_offset_y)
        
        # Apply transformations
        transformed_surface = pygame.transform.rotozoom(surface, self.rotation, self.scale)
        rect = transformed_surface.get_rect(center=adjusted_pos)
        
        # For proposals, check if point is within a certain radius of the center
        if is_proposal:
            # Calculate distance from point to center
            dx = point[0] - adjusted_pos[0]
            dy = point[1] - adjusted_pos[1]
            distance = (dx*dx + dy*dy) ** 0.5
            
            # Use a radius based on the surface size
            radius = min(transformed_surface.get_width(), transformed_surface.get_height()) * 0.4
            
            # If this is a proposal and it's animating, scale the radius
            if is_proposal and self.temporary_proposals[img_id].is_animating:
                radius *= self.temporary_proposals[img_id].animation_scale
                
            return distance <= radius
        else:
            # For regular images, use the rect collision
            screen_point = (point[0] + self.view_offset_x, point[1] + self.view_offset_y)
            return rect.collidepoint(screen_point)
        
    def distance_to_image(self, point, img_id, is_proposal=False):
        """Calculate the distance from a point to the center of an image"""
        # Get the appropriate image object based on whether it's a proposal or not
        if is_proposal:
            img_obj = self.temporary_proposals[img_id]
        else:
            img_obj = self.images[img_id]
            
        # Get the surface and position
        surface = img_obj.get_surface()
        pos = img_obj.get_pos()
        
        # Apply transformations to get the transformed surface
        transformed_surface = pygame.transform.rotozoom(surface, self.rotation, self.scale)
        
        # Apply view offset to position
        adjusted_pos = (pos[0] + self.view_offset_x, pos[1] + self.view_offset_y)
        
        # Get the rect of the transformed surface centered at the adjusted position
        rect = transformed_surface.get_rect(center=adjusted_pos)
        
        # Get the center of the rect (which is the center of the image)
        image_center = rect.center
        
        # Calculate distance between point and image center
        dx = point[0] - image_center[0]
        dy = point[1] - image_center[1]
        distance = (dx*dx + dy*dy) ** 0.5
        
        # Scale the distance by the current zoom level to maintain consistent visual distance
        #distance = distance * self.scale
        
        return distance

    def render_connections(self):
        """Draw white lines between related images"""
        # Draw connections for permanent images
        for img_id, img in self.images.items():
            parent_id = img.get_parent_id()
            if parent_id is not None and parent_id in self.images:
                # Get positions of both images
                parent_pos = self.images[parent_id].get_pos()
                child_pos = img.get_pos()
                
                # Apply view offset to positions
                parent_pos = (parent_pos[0] + self.view_offset_x, parent_pos[1] + self.view_offset_y)
                child_pos = (child_pos[0] + self.view_offset_x, child_pos[1] + self.view_offset_y)
                
                # Draw a white line between the images
                pygame.draw.line(self.screen, (255, 255, 255), parent_pos, child_pos, 2)
        
        # Draw connections for temporary proposals
        for prop in self.temporary_proposals.values():
            parent_id = prop.get_parent_id()
            if parent_id is not None and parent_id in self.images:
                # Get positions of both images
                parent_pos = self.images[parent_id].get_pos()
                child_pos = prop.get_pos()
                
                # Apply view offset to positions
                parent_pos = (parent_pos[0] + self.view_offset_x, parent_pos[1] + self.view_offset_y)
                child_pos = (child_pos[0] + self.view_offset_x, child_pos[1] + self.view_offset_y)
                
                # Draw a white line between the images
                pygame.draw.line(self.screen, (255, 255, 255), parent_pos, child_pos, 2)

    def start_relaxation(self):
        """Start the relaxation process for all images."""
        self.relaxing = True
        
    def get_image_depth(self, img_id, visited=None):
        """Calculate how many steps an image is from the original image.
        
        Args:
            img_id: ID of the image to check
            visited: Set of already visited image IDs to prevent infinite recursion
            
        Returns:
            int: Number of steps from the original image (0 for original, 1 for direct children, etc.)
        """
        # Initialize visited set if not provided
        if visited is None:
            visited = set()
        
        # If we've already visited this image, return a large number to prevent infinite recursion
        if img_id in visited:
            return 999999
        
        # Mark this image as visited
        visited.add(img_id)
        
        # If this is the original image (id 0), return 0
        if img_id == 0:
            return 0
        
        # Get the parent image
        parent_id = self.images[img_id].get_parent_id()
        if parent_id is not None and parent_id in self.images:
            # Recursively get the parent's depth and add 1
            return self.get_image_depth(parent_id, visited) + 1
        
        # If no parent found (shouldn't happen), return a large number
        return 999999

    def update_relaxation(self):
        """Update positions of all images with momentum-based movement and delayed following."""
        # Handle proposal animations
        current_time = pygame.time.get_ticks() / 1000.0  # Convert to seconds
        
        # Check if we should start showing proposals
        if self.is_animating_proposals and self.proposal_queue and not self.temporary_proposals:
            self.last_proposal_time = current_time
            # Add the first proposal immediately
            proposal = self.proposal_queue.pop(0)
            proposal_id = len(self.temporary_proposals)
            proposal.is_animating = True
            proposal.animation_start_time = current_time
            self.temporary_proposals[proposal_id] = proposal
        
        # Check if we should add the next proposal
        if self.is_animating_proposals and self.proposal_queue and current_time - self.last_proposal_time >= self.proposal_delay:
            # Add the next proposal to temporary_proposals
            proposal = self.proposal_queue.pop(0)
            proposal_id = len(self.temporary_proposals)
            proposal.is_animating = True
            proposal.animation_start_time = current_time
            self.temporary_proposals[proposal_id] = proposal
            self.last_proposal_time = current_time
            
            # If this was the last proposal, clear the queue
            if not self.proposal_queue:
                self.is_animating_proposals = False
        
        # Update animation states for all proposals
        for proposal in self.temporary_proposals.values():
            if proposal.is_animating:
                elapsed = current_time - proposal.animation_start_time
                if elapsed >= proposal.animation_duration:
                    proposal.is_animating = False
                    proposal.animation_scale = 1.0
                else:
                    # Use a smooth easing function for the animation
                    t = elapsed / proposal.animation_duration
                    proposal.animation_scale = t * t * (3 - 2 * t)  # Smooth step interpolation
        
        # Only relax if we're not showing proposals
        if self.temporary_proposals:
            return
            
        # Scale the minimum distance with the current zoom level
        scaled_min_distance = self.min_distance * self.scale
            
        # Calculate forces and update positions for all images
        for img_id1, img1 in self.images.items():
            pos1 = img1.get_pos()
            force_x = 0
            force_y = 0
            
            # Calculate repulsive forces from other images
            for img_id2, img2 in self.images.items():
                if img_id1 != img_id2:
                    pos2 = img2.get_pos()
                    dx = pos1[0] - pos2[0]
                    dy = pos1[1] - pos2[1]
                    distance = (dx*dx + dy*dy) ** 0.5
                    
                    if distance < scaled_min_distance:
                        # Calculate repulsive force with a stronger effect at closer distances
                        force = (scaled_min_distance - distance) * self.relaxation_strength * (1 + 1/distance)
                        if distance > 0:  # Avoid division by zero
                            force_x += (dx / distance) * force
                            force_y += (dy / distance) * force
            
            # Add gentle force based on image depth
            depth = self.get_image_depth(img_id1)
            if depth == 0:  # Original image
                # Calculate total rightward force from all other images
                total_right_force = 0
                for other_id, other_img in self.images.items():
                    other_depth = self.get_image_depth(other_id)
                    if other_depth > 0:  # Only count processed images
                        total_right_force += 0.1 * other_depth
                # Pull to the left with the same total force as all other images pull right
                force_x -= total_right_force
            else:  # Processed images
                # Pull to the right, more for deeper images
                force_x += 0.1 * depth
            
            # Get current velocity
            vx, vy = img1.get_velocity()
            
            # Add forces to velocity with increased responsiveness
            vx += force_x * self.relaxation_speed * 2
            vy += force_y * self.relaxation_speed * 2
            
            # Apply momentum (gradual slowdown)
            vx *= img1.momentum
            vy *= img1.momentum
            
            # Update velocity
            img1.set_velocity(vx, vy)
            
            # Calculate new position based on velocity
            new_x = pos1[0] + vx
            new_y = pos1[1] + vy
            
            # Keep images within bounds with a bounce effect
            if new_x < 50:
                new_x = 50
                vx = abs(vx) * 0.5  # Bounce with reduced velocity
            elif new_x > self.width - 50:
                new_x = self.width - 50
                vx = -abs(vx) * 0.5  # Bounce with reduced velocity
                
            if new_y < 50:
                new_y = 50
                vy = abs(vy) * 0.5  # Bounce with reduced velocity
            elif new_y > self.height - 50:
                new_y = self.height - 50
                vy = -abs(vy) * 0.5  # Bounce with reduced velocity
            
            # Update position
            img1.set_pos((new_x, new_y))
            
            # Update velocity after position update
            img1.set_velocity(vx, vy)
            
            # Update related images with delay and position preference
            related_ids = self.get_related_images(img_id1)
            for related_id in related_ids:
                if related_id != img_id1:
                    related_img = self.images[related_id]
                    
                    # Get current positions
                    current_pos = related_img.get_pos()
                    target_pos = img1.get_pos()
                    
                    # Calculate direction to target
                    dx = target_pos[0] - current_pos[0]
                    dy = target_pos[1] - current_pos[1]
                    
                    # Apply delayed following with position preference
                    follow_delay = related_img.get_follow_delay()
                    # Blend between current position and target position based on preferences
                    new_x = current_pos[0] * self.related_position_preference + (current_pos[0] + dx * follow_delay) * (1 - self.related_position_preference)
                    new_y = current_pos[1] * self.related_position_preference + (current_pos[1] + dy * follow_delay) * (1 - self.related_position_preference)
                    
                    # Keep within bounds
                    new_x = max(50, min(self.width - 50, new_x))
                    new_y = max(50, min(self.height - 50, new_y))
                    
                    # Update position
                    related_img.set_pos((new_x, new_y))
                    
                    # Gradually increase follow delay (slowing down)
                    related_img.set_follow_delay(follow_delay * 0.99)

    def get_related_images(self, img_id, visited=None):
        """Find all images related to the given image ID (children and parents).
        
        Args:
            img_id: ID of the image to find relations for
            visited: Set of already visited image IDs to prevent infinite recursion
            
        Returns:
            set: Set of image IDs that are related to the given image
        """
        # Initialize visited set if not provided
        if visited is None:
            visited = set()
        
        # If we've already visited this image, return empty set to prevent infinite recursion
        if img_id in visited:
            return set()
        
        # Mark this image as visited
        visited.add(img_id)
        
        related_ids = {img_id}  # Start with the image itself
        
        # Find all children (images that have this image as parent)
        for child_id, child in self.images.items():
            if child.get_parent_id() == img_id:
                # Recursively add children of children, passing the visited set
                child_related = self.get_related_images(child_id, visited)
                related_ids.update(child_related)
        
        # Find parent (if any)
        parent_id = self.images[img_id].get_parent_id()
        if parent_id is not None and parent_id in self.images:
            # Recursively add parents of parents, passing the visited set
            parent_related = self.get_related_images(parent_id, visited)
            related_ids.update(parent_related)
        
        return related_ids

    def move_image_group(self, img_id, new_pos):
        """Move an image and handle collisions with other images.
        
        Args:
            img_id: ID of the image to move
            new_pos: New position to move the image to
            
        Returns:
            bool: True if the move was successful
        """
        # Check if the image is at the border (with a small margin)
        border_margin = 10  # Pixels from the edge to consider as "border"
        
        # Convert world coordinates to screen coordinates for border check
        screen_pos = (new_pos[0] + self.view_offset_x, new_pos[1] + self.view_offset_y)
        
        at_border = (
            screen_pos[0] <= border_margin or 
            screen_pos[0] >= self.width - border_margin or
            screen_pos[1] <= border_margin or 
            screen_pos[1] >= self.height - border_margin
        )
        
        # If at border and not the original image, remove the image and its child nodes
        if at_border and img_id != 0:
            self.remove_image_and_children(img_id)
            return False
        
        # Keep the dragged image within bounds
        new_pos = (
            max(50, min(self.width - 50, new_pos[0])),
            max(50, min(self.height - 50, new_pos[1]))
        )
        
        # Scale the minimum distance with the current zoom level
        scaled_min_distance = self.min_distance * self.scale
        
        # Check for collisions with other images
        for other_id, other_img in self.images.items():
            if other_id != img_id:
                other_pos = other_img.get_pos()
                dx = new_pos[0] - other_pos[0]
                dy = new_pos[1] - other_pos[1]
                distance = (dx*dx + dy*dy) ** 0.5
                
                if distance < scaled_min_distance:
                    # Calculate push direction and distance
                    push_distance = scaled_min_distance - distance
                    if distance > 0:  # Avoid division by zero
                        push_x = (dx / distance) * push_distance
                        push_y = (dy / distance) * push_distance
                        
                        # Push the other image away
                        new_other_pos = (
                            other_pos[0] - push_x,
                            other_pos[1] - push_y
                        )
                        
                        # Keep within bounds
                        new_other_pos = (
                            max(50, min(self.width - 50, new_other_pos[0])),
                            max(50, min(self.height - 50, new_other_pos[1]))
                        )
                        
                        # Update other image's position
                        other_img.set_pos(new_other_pos)
                        
                        # Give the other image some velocity in the push direction
                        other_img.set_velocity(-push_x * 0.1, -push_y * 0.1)
        
        # Move the dragged image to its new position
        self.images[img_id].set_pos(new_pos)
        
        # Start relaxation to gradually adjust positions
        self.start_relaxation()
        
        return True
        
    def remove_image_and_children(self, img_id):
        """Remove an image and all its child nodes from the canvas.
        
        Args:
            img_id: ID of the image to remove
        """
        # Don't allow removing the original image (ID 0)
        if img_id == 0:
            return
            
        # Get all child images (not parents)
        child_ids = self.get_child_images(img_id)
        
        # Remove the image itself and all its children
        if img_id in self.images:
            del self.images[img_id]
            
        for child_id in child_ids:
            if child_id in self.images:
                del self.images[child_id]
                
        # Start relaxation to adjust remaining images
        self.start_relaxation()
        
    def get_child_images(self, img_id, visited=None):
        """Find all child images derived from the given image ID.
        
        Args:
            img_id: ID of the image to find children for
            visited: Set of already visited image IDs to prevent infinite recursion
            
        Returns:
            set: Set of image IDs that are children of the given image
        """
        # Initialize visited set if not provided
        if visited is None:
            visited = set()
        
        # If we've already visited this image, return empty set to prevent infinite recursion
        if img_id in visited:
            return set()
        
        # Mark this image as visited
        visited.add(img_id)
        
        child_ids = set()  # Start with empty set (don't include the image itself)
        
        # Find all direct children (images that have this image as parent)
        for child_id, child in self.images.items():
            if child.get_parent_id() == img_id:
                # Add this child
                child_ids.add(child_id)
                # Recursively add children of children, passing the visited set
                grandchild_ids = self.get_child_images(child_id, visited)
                child_ids.update(grandchild_ids)
        
        return child_ids


# Example image processing functions
def create_dummy_functions():
    """
    Create a collection of image processing functions organized by the type of image they can be applied to.
    Uses pyclesperanto for GPU-accelerated image processing.
    """
    # Initialize pyclesperanto
    cle.select_device('gpu')
    
    # Helper function to convert numpy array to pyclesperanto array
    def to_cle(array):
        return cle.push(array)
    
    # Helper function to convert pyclesperanto array to numpy array
    def to_numpy(array):
        return cle.pull(array)
    
    # Helper function to convert intensity to binary
    def intensity_to_binary(img, method='otsu'):
        if method == 'otsu':
            return to_numpy(cle.threshold_otsu(to_cle(img)))
        elif method == 'mean':
            # pyclesperanto doesn't have threshold_mean, so we'll implement it ourselves
            mean_val = np.mean(img)
            return (img > mean_val) * 1
        else:
            return (img > np.mean(img)) * 1
    
   
    
    # Functions for intensity images
    intensity_functions = {
        'denoise': {
            'gaussian': lambda img: to_numpy(cle.gaussian_blur(to_cle(img), sigma_x=2, sigma_y=2)),
            'median': lambda img: to_numpy(cle.median_box(to_cle(img), radius_x=1, radius_y=1)),
            'top_hat': lambda img: to_numpy(cle.top_hat_box(to_cle(img), radius_x=5, radius_y=5)),
            'bottom_hat': lambda img: to_numpy(cle.bottom_hat(to_cle(img), radius_x=5, radius_y=5)),
            'laplace': lambda img: to_numpy(cle.laplace(to_cle(img))),
            'laplace_of_gaussian': lambda img: to_numpy(cle.laplace_of_gaussian(to_cle(img), sigma_x=2, sigma_y=2)),
            'sobel': lambda img: to_numpy(cle.sobel(to_cle(img)))
        },
        'morphology': {
            'minimum': lambda img: to_numpy(cle.minimum_box(to_cle(img), radius_x=2, radius_y=2)),
            'maximum': lambda img: to_numpy(cle.maximum_box(to_cle(img), radius_x=2, radius_y=2)),
            'mean': lambda img: to_numpy(cle.mean_box(to_cle(img), radius_x=2, radius_y=2)),
            'variance': lambda img: to_numpy(cle.variance_box(to_cle(img), radius_x=2, radius_y=2)),
            'mode': lambda img: to_numpy(cle.mode(to_cle(img), radius_x=2, radius_y=2))
        },
        'threshold': {
            'otsu': lambda img: intensity_to_binary(img, 'otsu'),
            'mean': lambda img: intensity_to_binary(img, 'mean')
        },
        'segmentation': {
            'voronoi_otsu': lambda img: to_numpy(cle.voronoi_otsu_labeling(to_cle(img), spot_sigma=2, outline_sigma=2)),
            'eroded_otsu': lambda img: to_numpy(cle.eroded_otsu_labeling(to_cle(img), erosion_radius=2)),
            'gauss_otsu': lambda img: to_numpy(cle.gauss_otsu_labeling(to_cle(img), sigma_x=2, sigma_y=2)),
            'morphological_chan_vese': lambda img: to_numpy(cle.morphological_chan_vese(to_cle(img), iterations=10)),
            'voronoi': lambda img: to_numpy(cle.voronoi_labeling(to_cle(img)))
        },
        'background': {
            'subtract_gaussian': lambda img: to_numpy(cle.subtract_gaussian_background(to_cle(img), sigma_x=10, sigma_y=10)),
            'divide_by_gaussian': lambda img: to_numpy(cle.divide_by_gaussian_background(to_cle(img), sigma_x=10, sigma_y=10))
        }
    }
    
    # Functions for binary images
    binary_functions = {
        'morphology': {
            'dilate': lambda img: to_numpy(cle.maximum_box(to_cle(img), radius_x=2, radius_y=2)),
            'erode': lambda img: to_numpy(cle.minimum_box(to_cle(img), radius_x=2, radius_y=2)),
            'open': lambda img: to_numpy(cle.opening_box(to_cle(img), radius_x=2, radius_y=2)),
            'close': lambda img: to_numpy(cle.closing_box(to_cle(img), radius_x=2, radius_y=2)),
            'binary_not': lambda img: to_numpy(cle.binary_not(to_cle(img))),
            'binary_edge': lambda img: to_numpy(cle.binary_edge_detection(to_cle(img)))
        },
        'label': {
            'connected_components': lambda img: to_numpy(cle.connected_component_labeling(to_cle(img))).astype(np.int32),
        }
    }
    
    # Functions for labeled images
    label_functions = {
        'morphology': {
            'erode_labels': lambda img: to_numpy(cle.erode_labels(to_cle(img), radius=2)),
            'dilate_labels': lambda img: to_numpy(cle.dilate_labels(to_cle(img), radius=2)),
            'smooth_labels': lambda img: to_numpy(cle.smooth_labels(to_cle(img), radius=2)),
            'extend_via_voronoi': lambda img: to_numpy(cle.extend_labeling_via_voronoi(to_cle(img)))
        },
        'reduce': {
            'centroids': lambda img: to_numpy(cle.reduce_labels_to_centroids(to_cle(img))),
            'outlines': lambda img: to_numpy(cle.reduce_labels_to_label_edges(to_cle(img)))
        },
        'filter': {
            'remove_small': lambda img: to_numpy(cle.remove_small_labels(to_cle(img), minimum_size=100)),
            'remove_large': lambda img: to_numpy(cle.remove_large_labels(to_cle(img), maximum_size=1000))
        },
        'measure': {
            'mean_extension': lambda img: to_numpy(cle.mean_extension_map(to_cle(img))),
            'pixel_count': lambda img: to_numpy(cle.pixel_count_map(to_cle(img))),
            'extension_ratio': lambda img: to_numpy(cle.extension_ratio_map(to_cle(img)))
        }
    }
    
    # Combine all functions
    functions = {
        'intensity': intensity_functions,
        'binary': binary_functions,
        'label': label_functions,
        
    }
    
    return functions

# Demo
if __name__ == "__main__":
    # Load the human_mitosis_small.png image
    image_path = "human_mitosis_small.png"
    # Load image using matplotlib (handles various image formats well)
    img = mpimg.imread(image_path)
    # Convert to grayscale if it's a color image
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
    
    # Ensure the image is in the correct format for pyclesperanto (float32)
    img = img.astype(np.float32)
    
    # Create canvas with our enhanced function collection
    canvas = ImageProcessingCanvas(800, 600, create_dummy_functions())
    
    # Add initial image with numpy array
    canvas.add_image(img)
    
    # Run
    canvas.run()
