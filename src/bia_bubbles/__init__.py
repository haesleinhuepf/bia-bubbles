__version__ = "0.1.0"

import pygame
import numpy as np
from math import cos, sin, pi, atan2, dist
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import pyclesperanto as cle
import yaml
import os
import datetime
from scipy import ndimage as ndi
from skimage.morphology import binary_opening as sk_binary_opening
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.filters import gaussian, sobel
from skimage.morphology import label
from skimage.morphology import local_minima

class Connection:
    """A class that represents a connection between two images."""
    
    def __init__(self, start_img, end_img):
        """Initialize a connection between two images.
        
        Args:
            start_img: The starting image
            end_img: The ending image
        """
        self.start_img = start_img
        self.end_img = end_img
        
    def render(self, surface, scale=1.0):
        """Render the connection on the given surface.
        
        Args:
            surface: The pygame surface to render on
            scale: The scale factor to apply to the connection
        """
        # Get the positions of the two images
        start_pos = self.start_img.get_pos()
        end_pos = self.end_img.get_pos()
        
        # Draw a line between the two images
        pygame.draw.line(surface, (255, 255, 255), start_pos, end_pos, max(1, int(2 * scale)))

class ImageScatter:
    """A class that represents an image on the canvas with both numpy array and pygame surface representations."""
    
    def __init__(self, array, pos=None, parent_id=None, function_name=None, image_type=None, name=None, filename=None):
        """Initialize an ImageScatter with a numpy array.
        
        Args:
            array: Numpy array representation of the image
            pos: Position to place the image at (if None, will be centered)
            parent_id: ID of the parent image (if this is a processed result)
            function_name: Name of the function used to process the image
            image_type: Type of the image ('intensity', 'binary', or 'label')
            name: Display name of the image
            filename: Original filename of the image (if loaded from file)
        """
        self.array = array
        self.pos = pos
        self.parent_id = parent_id
        self.function_name = function_name
        self.image_type = image_type
        self.name = name
        self.filename = filename
        
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
            
        self.surface = pygame.surfarray.make_surface(display_array)
        self.radius = 100
        self.min_distance = 100
        
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

    def get_filename(self):
        """Get the original filename of the image."""
        return self.filename
        
    def set_filename(self, filename):
        """Set the original filename of the image."""
        self.filename = filename

    def render(self, surface, scale=1.0):
        """Render this image on the given surface.
        
        Args:
            surface: The pygame surface to render on
            scale: The scale factor to apply to the image
        """
        # Get the current position
        pos = self.get_pos()
        
        # Apply transformations
        transformed_surface = pygame.transform.rotozoom(self.surface, 0, scale)  # Use the provided scale
        
        # Create a circular mask
        mask_surface = pygame.Surface(transformed_surface.get_size(), pygame.SRCALPHA)
        self.radius = min(transformed_surface.get_width(), transformed_surface.get_height()) * 0.4

        # If this image is animating, scale the radius
        if self.is_animating:
            self.radius *= self.animation_scale
        
        center = (transformed_surface.get_width() // 2, transformed_surface.get_height() // 2)
        pygame.draw.circle(mask_surface, (255, 255, 255, 255), center, self.radius)
        
        # Create a temporary surface for the masked image
        temp_surface = pygame.Surface(transformed_surface.get_size(), pygame.SRCALPHA)
        temp_surface.blit(transformed_surface, (0, 0))
        
        # Apply the mask
        temp_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        
        # Get the rect for the masked image
        rect = temp_surface.get_rect(center=pos)
        
        # Render the masked image
        surface.blit(temp_surface, rect)
        
        # Draw white outline around the circle
        pygame.draw.circle(surface, (255, 255, 255), pos, self.radius, 2)
        
        # Render name if provided
        if self.name:
            self.render_name(surface, pos, self.radius, is_proposal=False, scale=scale)
        elif self.function_name:
            self.render_name(surface, pos, self.radius, is_proposal=True, scale=scale)

    def render_name(self, surface, pos, radius, is_proposal=False, scale=1.0):
        """Render the name of this image.
        
        Args:
            surface: The pygame surface to render on
            pos: Position of the image
            radius: Radius of the image circle
            is_proposal: Whether this is a proposal name (True) or an image name (False)
            scale: The scale factor to apply to the font size
        """
        # Create a font for the name with scaled size
        base_font_size = 24
        scaled_font_size = int(base_font_size * scale)
        font = pygame.font.Font(None, scaled_font_size)
        
        # Get the name to display
        name = self.function_name if is_proposal else self.name
        
        # Extract just the name part after the "/"
        display_name = name.split('/')[-1] if '/' in name else name
        
        # Split by "_" for multi-line display
        lines = display_name.split('_')
        
        # Calculate total height of all lines
        line_height = font.get_linesize()
        total_height = line_height * len(lines)
        
        # Calculate starting y position to center all lines vertically
        start_y = pos[1] - (total_height - line_height) / 2
        
        # Render each line
        for i, line in enumerate(lines):
            # Create text surfaces for this line
            text_white = font.render(line, True, (255, 255, 255))
            text_black = font.render(line, True, (0, 0, 0))
            
            # Calculate y position for this line
            line_y = start_y + i * line_height
            
            # Get the rects for all text surfaces
            text_rect_white_tl = text_white.get_rect(center=(pos[0] - 1, line_y - 1))  # Top-left
            text_rect_white_tr = text_white.get_rect(center=(pos[0] + 1, line_y - 1))  # Top-right
            text_rect_white_bl = text_white.get_rect(center=(pos[0] - 1, line_y + 1))  # Bottom-left
            text_rect_white_br = text_white.get_rect(center=(pos[0] + 1, line_y + 1))  # Bottom-right
            text_rect_black = text_black.get_rect(center=(pos[0], line_y))  # Center
            
            # Render all text surfaces for this line
            surface.blit(text_white, text_rect_white_tl)
            surface.blit(text_white, text_rect_white_tr)
            surface.blit(text_white, text_rect_white_bl)
            surface.blit(text_white, text_rect_white_br)
            surface.blit(text_black, text_rect_black)
    
    def screen_to_world(self, screen_pos):
        """Convert screen coordinates to world coordinates"""
        return screen_pos
        
    def point_in_image(self, point, img_id, is_proposal=False):
        """Check if a point is within an image's circular mask.
        
        Args:
            point: The point to check
            img_id: ID of the image to check
            is_proposal: Whether this is a proposal image
            
        Returns:
            bool: True if the point is within the image's circular mask
        """
        # Get the appropriate surface and position based on whether it's a proposal or not
        if is_proposal:
            img_obj = self.temporary_proposals[img_id]
        else:
            img_obj = self.images[img_id]
            
        # Get the position
        pos = img_obj.get_pos()
        
        # Calculate distance from point to center
        dx = point[0] - pos[0]
        dy = point[1] - pos[1]
        distance = (dx*dx + dy*dy) ** 0.5
        
        # Get the radius of the image's circular mask
        radius = img_obj.get_radius(self.scale)
        
        # Check if the point is within the radius
        return distance <= radius
        
    def distance_to_image(self, point, img_id, is_proposal=False):
        """Calculate the distance from a point to the edge of an image's circular mask.
        
        Args:
            point: The point to check
            img_id: ID of the image to check
            is_proposal: Whether this is a proposal image
            
        Returns:
            float: The distance from the point to the edge of the image's circular mask
        """
        # Get the appropriate image object based on whether it's a proposal or not
        if is_proposal:
            img_obj = self.temporary_proposals[img_id]
        else:
            img_obj = self.images[img_id]
            
        # Get the position
        pos = img_obj.get_pos()
        
        # Calculate distance from point to center
        dx = point[0] - pos[0]
        dy = point[1] - pos[1]
        distance = (dx*dx + dy*dy) ** 0.5
        
        # Get the radius of the image's circular mask
        radius = img_obj.get_radius(self.scale)
        
        # Return the distance to the edge of the circular mask
        return max(0, distance - radius)

    def get_radius(self, scale=1.0):
        """Get the radius of this image's circular mask.
        
        Args:
            scale: The scale factor to apply to the radius
            
        Returns:
            float: The radius of the circular mask
        """
        # Get the base radius from the surface size
        radius = min(self.surface.get_width(), self.surface.get_height()) * 0.4
        
        # Apply scale
        radius *= scale
        
        # If this image is animating, scale the radius
        if self.is_animating:
            radius *= self.animation_scale
            
        return radius


class ImageProcessingCanvas:
    def __init__(self, width=800, height=600, function_collection=None):
        pygame.init()
        # Set window title
        pygame.display.set_caption("Bio-image Analysis in Bubbles - BIA-bubbles")
        # Create resizable window with drag and drop support
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE | pygame.DROPFILE)
        self.clock = pygame.time.Clock()
        
        self.images = {}  # {id: ImageScatter}
        self.connections = []
        self.temporary_proposals = {}  # {id: ImageScatter}
        
        # Add title property
        self.title = None
        self.title_font = pygame.font.Font(None, 36)  # Larger font for title
        
        # Add current level tracking
        self.current_level = 0
        
        # Add save button properties
        self.save_button_rect = pygame.Rect(10, height - 40, 100, 30)
        self.save_button_color = (196, 196, 196) 
        self.save_button_hover_color = (128, 128, 128)  
        self.save_button_text = "Save"
        self.save_button_font = pygame.font.Font(None, 24)
        self.save_button_hovered = False
        
        # Add level button properties
        self.level_button_rect = pygame.Rect(120, height - 40, 100, 30)  # Positioned next to save button
        self.level_button_color = (196, 196, 196)
        self.level_button_hover_color = (128, 128, 128)
        self.level_button_text = "Level"
        self.level_button_hovered = False
        self.level_buttons_visible = False
        self.level_buttons = []
        self.level_button_height = 30
        self.level_button_spacing = 5
        
        # Add solution button properties
        self.solution_button_rect = pygame.Rect(230, height - 40, 100, 30)  # Positioned next to level button
        self.solution_button_color = (196, 196, 196)
        self.solution_button_hover_color = (128, 128, 128)
        self.solution_button_text = "Solution"
        self.solution_button_hovered = False
        
        # Add quality metric properties
        self.quality_metric = None
        self.quality_metric_font = pygame.font.Font(None, 24)
        self.quality_metric_rect = pygame.Rect(340, height - 40, 200, 30)  # Positioned next to solution button
        
        # Add level completion properties
        self.level_completed = False
        self.level_completion_time = 0
        self.level_completion_delay = 5000  # 5 seconds in milliseconds
        self.correct_font = pygame.font.Font(None, 72)  # Large font for "Correct!" message
        
        # Find max level number by checking for level{n}.yaml files
        max_level = 1
        while os.path.exists(f"level{max_level}.yaml"):
            max_level += 1
        max_level -= 1  # Adjust since we went one too far
        
        # Create level selection buttons (1 to max_level)
        for i in range(max_level):
            button_rect = pygame.Rect(
                self.level_button_rect.x,
                self.level_button_rect.y - (i + 1) * (self.level_button_height + self.level_button_spacing),
                self.level_button_rect.width,
                self.level_button_height
            )
            self.level_buttons.append(button_rect)
        
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
        self.function_filters = None
        
    def add_image(self, array, parent_id=None, pos=None, image_type=None, filename=None, name=None):
        """Add a new image to the canvas.
        
        Args:
            array: Numpy array representation of the image
            parent_id: ID of the parent image (if this is a processed result)
            pos: Position to place the image at (if None, will be centered)
            image_type: Type of the image ('intensity', 'binary', or 'label'). If None, will be determined automatically.
            filename: Original filename of the image (if loaded from file)
            name: Display name of the image
        """
        if len(self.images) == 0:
            img_id = 0
        else:
            img_id = int(np.max(list(self.images.keys()))) + 1
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
        if name is None:
            if img_id == 0:
                name = "original"
            elif parent_id is not None and parent_id in self.images:
                # For processed images, use the function name
                parent_img = self.images[parent_id]
                if parent_img.get_function_name():
                    name = parent_img.get_function_name()
            
        image_scatter = ImageScatter(array, pos, parent_id, image_type=image_type, name=name, filename=filename)
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
        # Don't process result images
        if self.is_result_image(img_id):
            return
            
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
            type_functions = filter_functions(self.function_collection[img_type], self.function_filters)
            
            # Calculate total number of proposals
            total_functions = sum(len(funcs) for funcs in type_functions.values())
            
            # Determine initial distance based on number of proposals
            # More proposals need more space
            factor = 2.2
            if total_functions > 5:
                factor = 2.5
            if total_functions > 10:
                factor = 3
            if total_functions > 15:
                factor = 4
            base_distance = self.images[img_id].get_radius(self.scale) * factor
                
            # Scale the base distance by the current zoom level to maintain consistent visual distance
            #base_distance = base_distance * self.scale
                
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
        if function_name == 'minimum':
            return input_type
        if function_name == 'maximum':
            return input_type
            
        # Threshold functions convert intensity to binary
        if input_type == 'intensity' and category == 'binarization':
            return 'binary'
        if input_type == 'intensity' and category == 'label':
            return 'label'


        # Connected components convert binary to label
        if input_type == 'binary' and function_name == 'connected_components':
            return 'label'
        if input_type == 'binary' and function_name == 'label_spots':
            return 'label'
        if input_type == 'binary' and function_name == 'distance_map':
            return 'intensity'
            
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
                # Calculate scale factors based on size change
                width_scale = event.w / self.width
                height_scale = event.h / self.height
                # Use the smaller scale factor to ensure everything fits
                scale_factor = min(width_scale, height_scale)
                
                # Update window size
                self.width, self.height = event.size
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                
                # Update button positions
                self.save_button_rect.bottom = self.height - 10
                self.level_button_rect.bottom = self.height - 10
                self.solution_button_rect.bottom = self.height - 10
                self.quality_metric_rect.bottom = self.height - 10
                
                # Update level selection buttons
                for i, button_rect in enumerate(self.level_buttons):
                    button_rect.x = self.level_button_rect.x
                    button_rect.y = self.level_button_rect.y - (i + 1) * (self.level_button_height + self.level_button_spacing)
                
                # Scale the scene by applying zoom at the center of the window
                center_point = (0,0)
                self.zoom_at_point(center_point, scale_factor)
            
            # Handle drag and drop
            elif event.type == pygame.DROPFILE:
                filepath = event.file
                # Check if it's a YAML file
                if filepath.lower().endswith('.yaml') or filepath.lower().endswith('.yml'):
                    self.load_yaml_tree(filepath)
                else:
                    # Try to load as an image
                    self.load_image(filepath)
            
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
                    # Check if level button was clicked
                    if self.handle_level_button_click(event.pos):
                        continue

                    if self.handle_solution_button_click(event.pos):
                        continue
                        
                    # Check if save button was clicked
                    if self.save_button_rect.collidepoint(event.pos):
                        self.save_image_tree()
                        # Don't return here, just continue with the rest of the method
                    
                    # Check temporary proposals first
                    closest_proposal_id = None
                    closest_distance = float('inf')
                    
                    for prop_id, prop in self.temporary_proposals.items():
                        if self.point_in_image(event.pos, prop_id, is_proposal=True):
                            # Calculate distance to this proposal
                            distance = self.distance_to_image(event.pos, prop_id, is_proposal=True)
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
                            
                        # Compute quality metric if there's a result image
                        result_id = None
                        for img_id, img in self.images.items():
                            if self.is_result_image(img_id):
                                result_id = img_id
                                break
                                
                        if result_id is not None:
                            result_img = self.images[result_id]
                            result_type = result_img.get_image_type()
                            
                            # Only compute metric if types match
                            if result_type == proposal_type:
                                self.quality_metric = self.compute_quality_metric(
                                    result_img.get_array(),
                                    self.temporary_proposals[closest_proposal_id].get_array(),
                                    result_type,
                                    proposal_type
                                )
                                # Check if we should load next level
                                self.check_quality_and_load_next_level()

                                if self.level_completed:
                                    self.images[result_id].parent_id = new_img_id
                                    print(f"Connected result image to {new_img_id}")
                            else:
                                self.quality_metric = None
                        else:
                            self.quality_metric = None
                            
                        self.temporary_proposals.clear()
                        # Start relaxation after adding a new image
                        self.start_relaxation()
                    # Then check permanent images
                    else:
                        image_clicked = False
                        for img_id in self.images:
                            # Skip result images
                            if self.is_result_image(img_id):
                                continue
                                
                            if self.point_in_image(event.pos, img_id):
                                # Store click information for potential drag or click
                                self.click_start_time = pygame.time.get_ticks()
                                self.click_start_pos = event.pos
                                self.clicked_image_id = img_id
                                # Calculate offset from mouse to image center for potential drag
                                img_pos = self.images[img_id].get_pos()
                                self.drag_offset = (img_pos[0] - event.pos[0], img_pos[1] - event.pos[1])
                                image_clicked = True
                                break
                        
                        # If no image was clicked and we have proposals, clear them and start panning
                        if not image_clicked and self.temporary_proposals:
                            self.temporary_proposals.clear()
                            self.start_relaxation()
                            self.panning = True
                            self.pan_start = event.pos
                        # If no image was clicked and no proposals, just start panning
                        elif not image_clicked:
                            self.panning = True
                            self.pan_start = event.pos
                
                elif event.button == 3:  # Right click
                    # Check if we clicked on an image
                    if self.clicked_image_id is not None and self.clicked_image_id in self.images:
                        # Skip result images
                        if self.is_result_image(self.clicked_image_id):
                            return True
                            
                        # If we clicked on an image, handle rotation
                        self.dragging = True
                        self.drag_start = event.pos
                        self.initial_rotation = self.rotation
                    else:
                        # If we didn't click on an image, handle panning
                        self.panning = True
                        self.pan_start = event.pos
                
                elif event.button == 2:  # Middle click
                    pass  # Disabled middle click panning
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left click
                    # Check if this was a click (short press) or a drag
                    if self.clicked_image_id is not None and self.clicked_image_id in self.images:
                        # Skip result images
                        if self.is_result_image(self.clicked_image_id):
                            return True
                            
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
                    pass  # Disabled middle click panning
            
            elif event.type == pygame.MOUSEMOTION:
                # Check if left mouse button is pressed
                if pygame.mouse.get_pressed()[0]:  # Left button
                    # Check if we should start dragging
                    if self.clicked_image_id is not None and not self.dragging:
                        # Skip result images
                        if self.is_result_image(self.clicked_image_id):
                            return True
                            
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
                        # Skip result images
                        if self.is_result_image(self.dragged_image_id):
                            return True
                            
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
                # Check if middle mouse button is pressed
                elif pygame.mouse.get_pressed()[1]:  # Middle button
                    pass  # Disabled middle button panning
                # Check if right mouse button is pressed
                elif pygame.mouse.get_pressed()[2]:  # Right button
                    if self.dragging:
                        # Skip result images
                        if self.clicked_image_id is not None and self.is_result_image(self.clicked_image_id):
                            return True
                            
                        # Handle rotation
                        dx = event.pos[0] - self.drag_start[0]
                        self.rotation = self.initial_rotation + dx * 0.01
            
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
                    self.initial_touch_distance = dist(*self.touch_points)
                    self.initial_touch_angle = atan2(self.touch_points[1][1] - self.touch_points[0][1],
                                                   self.touch_points[1][0] - self.touch_points[0][0])
            
            elif event.type == pygame.FINGERUP:
                # Remove the finger that was lifted
                finger_id = event.finger_id
                if finger_id < len(self.touch_points):
                    self.touch_points.pop(finger_id)
                
                # Update touch states based on remaining fingers
                if len(self.touch_points) == 0:
                    # No fingers left - disable multi-touch
                    self.multi_touch = False
                elif len(self.touch_points) == 1:
                    # One finger left - disable multi-touch
                    self.multi_touch = False
            
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
        
        # Add permanent images (excluding result image)
        for img_id, img in self.images.items():
            if not self.is_result_image(img_id):
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
            
        # If we have proposals, ensure they maintain the same distance from their parent
        if self.temporary_proposals:
            # Get the parent image ID from any proposal (they all have the same parent)
            parent_id = next(iter(self.temporary_proposals.values())).get_parent_id()
            if parent_id in self.images:
                parent_pos = self.images[parent_id].get_pos()
                
                # Calculate the original distance from parent to each proposal
                for prop_id, prop in self.temporary_proposals.items():
                    prop_pos = prop.get_pos()
                    # Calculate vector from parent to proposal
                    dx = prop_pos[0] - parent_pos[0]
                    dy = prop_pos[1] - parent_pos[1]
                    # Calculate distance
                    distance = (dx*dx + dy*dy) ** 0.5
                    
                    # Calculate angle
                    angle = atan2(dy, dx)
                    
                    # Calculate new position maintaining the same angle but with adjusted distance
                    # We need to scale the distance by the zoom factor to maintain visual consistency
                    new_distance = distance * zoom_factor
                    new_x = parent_pos[0] + cos(angle) * new_distance
                    new_y = parent_pos[1] + sin(angle) * new_distance
                    
                    # Set new position
                    prop.set_pos((new_x, new_y))
                    
        # Move result image to bottom right corner after zooming
        self.move_result_to_bottom_right()

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
        """Render the canvas and all its elements."""
        # Clear the screen
        self.screen.fill((30, 30, 30))
        
        # Render the title if it exists
        if self.title:
            title_font = pygame.font.Font(None, 36)
            title_text = title_font.render(self.title, True, (255, 255, 255))
            title_rect = title_text.get_rect(center=(self.width // 2, 30))
            self.screen.blit(title_text, title_rect)
        
        # Render connections
        self.render_connections()
        
        # Render images
        for img_id, img in self.images.items():
            img.render(self.screen, self.scale)
            self.min_distance = img.radius / self.scale * 2.1
        
        # Render temporary proposals
        for prop_id, prop in self.temporary_proposals.items():
            prop.render(self.screen, self.scale)
        
        # Render the save button
        self.render_save_button()
        
        # Render the level buttons
        self.render_level_buttons()
        
        # Render the solution button
        self.render_solution_button()
        
        # Render quality metric
        self.render_quality_metric()
        
        # Render "Correct!" message if level is completed
        if self.level_completed:
            # Create a semi-transparent overlay
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))  # Semi-transparent black
            self.screen.blit(overlay, (0, 0))
            
            # Render "Correct!" text
            correct_text = self.correct_font.render("Correct!", True, (0, 255, 0))  # Green text
            correct_rect = correct_text.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(correct_text, correct_rect)
            

        # Update the display
        pygame.display.flip()

    def render_title(self, surface):
        """Render the title at the top of the screen."""
        if self.title:
            # Create a semi-transparent background for the title
            title_surface = self.title_font.render(self.title, True, (255, 255, 255))
            title_rect = title_surface.get_rect(center=(self.width // 2, 30))
            
            # Create a background rectangle for the title
            padding = 10
            bg_rect = pygame.Rect(
                title_rect.left - padding,
                title_rect.top - padding,
                title_rect.width + 2 * padding,
                title_rect.height + 2 * padding
            )
            
            # Draw semi-transparent background
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            bg_surface.fill((0, 0, 0, 128))  # Black with 50% transparency
            surface.blit(bg_surface, bg_rect)
            
            # Draw the title text
            surface.blit(title_surface, title_rect)

    def render_save_button(self):
        """Render the save button on the screen."""
        button_color = self.save_button_hover_color if self.save_button_hovered else self.save_button_color
        pygame.draw.rect(self.screen, button_color, self.save_button_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), self.save_button_rect, 2)
        
        text_surface = self.save_button_font.render(self.save_button_text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.save_button_rect.center)
        self.screen.blit(text_surface, text_rect)

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
    
    def render_name(self, pos, name, radius, is_proposal=False, scale=1.0):
        """Render a name (either function name for proposals or image name) with appropriate styling.
        
        Args:
            pos: Position of the image
            name: Name to render
            radius: Radius of the image circle
            is_proposal: Whether this is a proposal name (True) or an image name (False)
            scale: The scale factor to apply to the font size
        """
        if name is None:
            return
            
        # Scale the radius with the current scale to maintain proper distance
        scaled_radius = radius * self.scale
        
        # Scale font size with the current scale
        base_font_size = 24
        scaled_font_size = int(base_font_size * scale)
        font = pygame.font.Font(None, scaled_font_size)
        
        # Extract just the name part after the "/"
        display_name = name.split('/')[-1] if '/' in name else name
        
        # For images, position text at the center with white outline
        # Position text at the center of the image
        text_x = pos[0]
        text_y = pos[1]
        
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
        return screen_pos
        
    def point_in_image(self, point, img_id, is_proposal=False):
        """Check if a point is within an image's circular mask.
        
        Args:
            point: The point to check
            img_id: ID of the image to check
            is_proposal: Whether this is a proposal image
            
        Returns:
            bool: True if the point is within the image's circular mask
        """
        # Get the appropriate surface and position based on whether it's a proposal or not
        if is_proposal:
            img_obj = self.temporary_proposals[img_id]
        else:
            img_obj = self.images[img_id]
            
        # Get the position
        pos = img_obj.get_pos()
        
        # Calculate distance from point to center
        dx = point[0] - pos[0]
        dy = point[1] - pos[1]
        distance = (dx*dx + dy*dy) ** 0.5
        
        # Get the radius of the image's circular mask
        radius = img_obj.get_radius(self.scale)
        
        # Check if the point is within the radius
        return distance <= radius
        
    def distance_to_image(self, point, img_id, is_proposal=False):
        """Calculate the distance from a point to the edge of an image's circular mask.
        
        Args:
            point: The point to check
            img_id: ID of the image to check
            is_proposal: Whether this is a proposal image
            
        Returns:
            float: The distance from the point to the edge of the image's circular mask
        """
        # Get the appropriate image object based on whether it's a proposal or not
        if is_proposal:
            img_obj = self.temporary_proposals[img_id]
        else:
            img_obj = self.images[img_id]
            
        # Get the position
        pos = img_obj.get_pos()
        
        # Calculate distance from point to center
        dx = point[0] - pos[0]
        dy = point[1] - pos[1]
        distance = (dx*dx + dy*dy) ** 0.5
        
        # Get the radius of the image's circular mask
        radius = img_obj.get_radius(self.scale)
        
        # Return the distance to the edge of the circular mask
        return max(0, distance - radius)

    def render_connections(self):
        """Draw white lines between related images"""
        # Draw connections for permanent images
        for img_id, img in self.images.items():
            parent_id = img.get_parent_id()
            if parent_id is not None and parent_id in self.images:
                # Get positions of both images
                parent_pos = self.images[parent_id].get_pos()
                child_pos = img.get_pos()
                
                # Create a connection object
                connection = Connection(self.images[parent_id], img)
                connection.render(self.screen, self.scale)
        
        # Draw connections for temporary proposals
        for prop in self.temporary_proposals.values():
            parent_id = prop.get_parent_id()
            if parent_id is not None and parent_id in self.images:
                # Get positions of both images
                parent_pos = self.images[parent_id].get_pos()
                child_pos = prop.get_pos()
                
                # Create a connection object
                connection = Connection(self.images[parent_id], prop)
                connection.render(self.screen, self.scale)

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
        
        # Update animation scale for all proposals
        for proposal in self.temporary_proposals.values():
            if proposal.is_animating:
                elapsed_time = current_time - proposal.animation_start_time
                # Calculate scale based on elapsed time (0 to 1 over animation_duration)
                proposal.animation_scale = min(1.0, elapsed_time / proposal.animation_duration)
                # If animation is complete, stop animating
                if proposal.animation_scale >= 1.0:
                    proposal.is_animating = False
                    proposal.animation_scale = 1.0
        
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
        
        # Check if level is completed and it's time to load the next level
        if self.level_completed:
            current_ticks = pygame.time.get_ticks()
            if current_ticks - self.level_completion_time >= self.level_completion_delay:
                self.level_completed = False
                self.load_next_level()
                return
        
        # Only relax if we're not showing proposals
        if self.temporary_proposals:
            return
            
        # Scale the minimum distance with the current zoom level
        scaled_min_distance = self.min_distance * self.scale
            
        # Calculate forces and update positions for all images
        for img_id1, img1 in self.images.items():
            # Skip result images
            if self.is_result_image(img_id1):
                continue
                
            pos1 = img1.get_pos()
            force_x = 0
            force_y = 0
            
            # Calculate repulsive forces from other images
            for img_id2, img2 in self.images.items():
                # Skip result images and self
                if img_id1 == img_id2 or self.is_result_image(img_id2):
                    continue
                    
                pos2 = img2.get_pos()
                dx = pos1[0] - pos2[0]
                dy = pos1[1] - pos2[1]
                distance = (dx*dx + dy*dy) ** 0.5
                if distance == 0:
                    distance = 0.001
                
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
                    # Skip result images
                    if self.is_result_image(other_id):
                        continue
                        
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
            if new_x < img2.radius:
                new_x = img2.radius
                vx = abs(vx) * 0.5  # Bounce with reduced velocity
            elif new_x > self.width - img2.radius:
                new_x = self.width - img2.radius
                vx = -abs(vx) * 0.5  # Bounce with reduced velocity
                
            if new_y < img2.radius:
                new_y = img2.radius
                vy = abs(vy) * 0.5  # Bounce with reduced velocity
            elif new_y > self.height - img2.radius:
                new_y = self.height - img2.radius
                vy = -abs(vy) * 0.5  # Bounce with reduced velocity
            
            # Update position
            img1.set_pos((new_x, new_y))
            
            # Update velocity after position update
            img1.set_velocity(vx, vy)
            
            # Update related images with delay and position preference
            related_ids = self.get_related_images(img_id1)
            for related_id in related_ids:
                # Skip result images and self
                if related_id == img_id1 or self.is_result_image(related_id):
                    continue
                    
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
                new_x = max(img2.radius, min(self.width - img2.radius, new_x))
                new_y = max(img2.radius, min(self.height - img2.radius, new_y))
                
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
        # Don't move result images
        if self.is_result_image(img_id):
            return False
            
        # Check if the image is at the border (with a small margin)
        border_margin = 10  # Pixels from the edge to consider as "border"
        
        at_border = (
            new_pos[0] <= border_margin or 
            new_pos[0] >= self.width - border_margin or
            new_pos[1] <= border_margin or 
            new_pos[1] >= self.height - border_margin
        )
        
        # If at border and not the original image, remove the image and its child nodes
        if at_border and img_id != 0:
            self.remove_image_and_children(img_id)
            return False
        
        img = self.images[img_id]
        
        # Keep the dragged image within bounds
        new_pos = (
            max(img.radius, min(self.width - img.radius, new_pos[0])),
            max(img.radius, min(self.height - img.radius, new_pos[1]))
        )
        print("min_distance", self.min_distance)
        
        # Scale the minimum distance with the current zoom level
        scaled_min_distance = self.min_distance * self.scale
        
        # Check for collisions with other images
        for other_id, other_img in self.images.items():
            # Skip result images and self
            if other_id == img_id or self.is_result_image(other_id):
                continue
                
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
                        max(img.radius, min(self.width - img.radius, new_other_pos[0])),
                        max(img.radius, min(self.height - img.radius, new_other_pos[1]))
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

    def draw(self):
        self.screen.fill((255, 255, 255))
        
        # Draw connections first
        for connection in self.connections:
            pygame.draw.line(self.screen, (0, 0, 0), connection[0], connection[1], 2)
        
        # Draw all images
        for image in self.images.values():
            image.draw(self.screen)
        
        # Draw temporary proposals
        for proposal in self.temporary_proposals.values():
            proposal.draw(self.screen)
            
        # Draw save button
        button_color = self.save_button_hover_color if self.save_button_hovered else self.save_button_color
        pygame.draw.rect(self.screen, button_color, self.save_button_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), self.save_button_rect, 2)
        
        text_surface = self.save_button_font.render(self.save_button_text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.save_button_rect.center)
        self.screen.blit(text_surface, text_rect)
        
        pygame.display.flip()

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                # Check if save button was clicked
                if self.save_button_rect.collidepoint(event.pos):
                    self.save_image_tree()
                    # Don't return here, just continue with the rest of the method
                    
                # Check temporary proposals first
                closest_proposal_id = None
                closest_distance = float('inf')
                
                for prop_id, prop in self.temporary_proposals.items():
                    if self.point_in_image(event.pos, prop_id, is_proposal=True):
                        # Calculate distance to this proposal
                        distance = self.distance_to_image(event.pos, prop_id, is_proposal=True)
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
                        # Skip result images
                        if self.is_result_image(img_id):
                            continue
                                
                        if self.point_in_image(event.pos, img_id):
                            # Store click information for potential drag or click
                            self.click_start_time = pygame.time.get_ticks()
                            self.click_start_pos = event.pos
                            self.clicked_image_id = img_id
                            # Calculate offset from mouse to image center for potential drag
                            img_pos = self.images[img_id].get_pos()
                            self.drag_offset = (img_pos[0] - event.pos[0], img_pos[1] - event.pos[1])
                            image_clicked = True
                            break
                        
                        # If no image was clicked and we have proposals, clear them and start panning
                        if not image_clicked and self.temporary_proposals:
                            self.temporary_proposals.clear()
                            self.start_relaxation()
                            self.panning = True
                            self.pan_start = event.pos
                        # If no image was clicked and no proposals, just start panning
                        elif not image_clicked:
                            self.panning = True
                            self.pan_start = event.pos
            
            elif event.button == 3:  # Right click
                # Check if we clicked on an image
                if self.clicked_image_id is not None and self.clicked_image_id in self.images:
                    # Skip result images
                    if self.is_result_image(self.clicked_image_id):
                        return True
                        
                    # If we clicked on an image, handle rotation
                    self.dragging = True
                    self.drag_start = event.pos
                    self.initial_rotation = self.rotation
                else:
                    # If we didn't click on an image, handle panning
                    self.panning = True
                    self.pan_start = event.pos
            
            elif event.button == 2:  # Middle click
                pass  # Disabled middle click panning
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left click
                # Check if this was a click (short press) or a drag
                if self.clicked_image_id is not None and self.clicked_image_id in self.images:
                    # Skip result images
                    if self.is_result_image(self.clicked_image_id):
                        return True
                        
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
                pass  # Disabled middle click panning
        
        elif event.type == pygame.MOUSEMOTION:
            # Update save button hover state
            self.save_button_hovered = self.save_button_rect.collidepoint(event.pos)
            # Update level button hover state
            self.level_button_hovered = self.level_button_rect.collidepoint(event.pos)
            # Update solution button hover state
            self.solution_button_hovered = self.solution_button_rect.collidepoint(event.pos)
            
            # Handle image dragging
            for image in self.images.values():
                if image.selected:
                    image.move(event.pos)
                    
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.temporary_proposals.clear()
                
        elif event.type == pygame.VIDEORESIZE:
            self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            # Update save button position
            self.save_button_rect.bottom = event.h - 10

    def save_image_tree(self):
        """Save the current image tree structure to a YAML file."""
        def build_tree_data(image_id):
            """Recursively build the tree data structure for an image and its children."""
            image = self.images[image_id]
            pos = image.get_pos()
            data = {
                'id': image_id,
                'name': image.get_name(),
                'type': image.get_image_type(),
                'x': pos[0],  # Save x coordinate separately
                'y': pos[1],  # Save y coordinate separately
                'filename': image.get_filename(),  # Add filename to the saved data
                'children': []
            }
            
            # Find all children of this image (only from permanent images)
            for child_id in self.images:
                child = self.images[child_id]
                if child.get_parent_id() == image_id:
                    data['children'].append(build_tree_data(child_id))
                    
            return data
            
        # Build the tree starting from the root (image with no parent)
        root_id = None
        for img_id, img in self.images.items():
            if img.get_parent_id() is None:
                root_id = img_id
                break
                
        if root_id is not None:
            tree_data = build_tree_data(root_id)
            
            # Add view settings to the tree data
            tree_data['view_settings'] = {
                'scale': self.scale,
                'rotation': self.rotation
            }
            
            # Create a timestamp for the filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_tree_{timestamp}.yaml"
            
            # Save to YAML file
            with open(filename, 'w') as f:
                yaml.dump(tree_data, f, default_flow_style=False)
                
    def load_image(self, filepath):
        """Load an image from a file and add it to the canvas.
        
        Args:
            filepath: Path to the image file to load
        """
        try:
            # Load the image using matplotlib
            img = mpimg.imread(filepath)
            
            # Convert to grayscale if it's RGB
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
                
            # Normalize to 0-1 range if needed
            if img.max() > 1.0:
                img = img / 255.0
                
            # Get just the filename without the path
            filename = os.path.basename(filepath)
            
            # clear canvas
            self.clear_canvas()
            self.function_filters = None
            self.title = ""

            # Add the image to the canvas
            self.add_image(img, filename=filename)
            
        except Exception as e:
            print(f"Error loading image: {e}")

    def clear_canvas(self):
        """Clear all images from the canvas."""
        self.images.clear()
        self.temporary_proposals.clear()
        self.proposal_queue.clear()
        self.is_animating_proposals = False
        self.start_relaxation()

    def load_yaml_tree(self, filepath, show_solution=False):
        """Load an image tree from a YAML file.
        
        Args:
            filepath: Path to the YAML file to load
        """
        try:
            # Clear the canvas first
            self.clear_canvas()
            
            # Load the YAML file
            with open(filepath, 'r') as f:
                tree_data = yaml.safe_load(f)

            # Extract function filters if present
            self.function_filters = tree_data.get('function_filters', None)
                
            # Extract title if present
            self.title = tree_data.get('title', None)
            
            # Extract level number from title if present
            if self.title and "Level" in self.title:
                try:
                    level_str = self.title.split("Level")[1].split(":")[0].strip()
                    self.current_level = int(level_str)
                except (IndexError, ValueError):
                    self.current_level = 0
            
            # Extract view settings if present
            view_settings = tree_data.get('view_settings', {})
            self.scale = view_settings.get('scale', 1.0)
            self.rotation = view_settings.get('rotation', 0)
            self.view_offset_x = view_settings.get('view_offset_x', 0)
            self.view_offset_y = view_settings.get('view_offset_y', 0)
                
            # Create a mapping of old IDs to new IDs
            id_mapping = {}
            
            # Calculate scaling factors based on original (800x600) and current screen size
            scale_x = self.width / 800.0
            scale_y = self.height / 600.0
            
            def load_node(node_data, parent_id=None):
                """Recursively load a node and its children."""
                # Get the image type and name
                image_type = node_data.get('type', 'intensity')
                name = node_data.get('name', None)
                
                # Scale the position based on current screen size
                x = node_data.get('x', 0) * scale_x
                y = node_data.get('y', 0) * scale_y
                
                # If this is a processed image (has a name with category/function format)
                if name and '/' in name:
                    category, function_name = name.split('/')
                    
                    # Try to find the function in any available image type
                    function_found = False
                    
                    # Loop through all available image types
                    for type_name, type_functions in self.function_collection.items():
                        # Check if the category and function exist in this image type
                        if category in type_functions and function_name in type_functions[category]:
                            # Get the parent image to process
                            if parent_id is not None and parent_id in self.images:
                                parent_img = self.images[parent_id]
                                # Process the parent image using the function
                                result_array = type_functions[category][function_name](parent_img.get_array())
                                # Add the processed image
                                new_id = self.add_image(
                                    result_array,
                                    parent_id=parent_id,
                                    pos=(x, y),
                                    image_type=image_type,  # Use the type specified in the YAML file
                                    name=name
                                )
                                # Store the ID mapping
                                id_mapping[node_data['id']] = new_id
                                
                                # Load children recursively
                                if 'children' in node_data:
                                    for child_data in node_data['children']:
                                        load_node(child_data, parent_id=new_id)
                                        
                                function_found = True
                                return new_id
                    
                    # If we didn't find the function in any image type
                    if not function_found:
                        if parent_id is not None and parent_id in self.images:
                            searched_types = list(self.function_collection.keys())
                            print(f"Error: Function '{category}/{function_name}' not found in any available image type")
                            print(f"Available image types: {', '.join(searched_types)}")
                            print(f"Please check if the function exists in one of these image types")
                        else:
                            print(f"Error: Cannot find parent image for processed image '{name}'")
                
                # If this is an original image (has a filename)
                elif 'filename' in node_data and node_data['filename']:
                    # Load the image file
                    img = mpimg.imread(os.path.join(os.path.dirname(filepath), node_data['filename']))
                    
                    # Convert to grayscale if it's RGB
                    if len(img.shape) == 3:
                        img = np.mean(img, axis=2)
                        
                    # Normalize to 0-1 range if needed
                    if img.max() > 1.0:
                        img = img / 255.0
                        
                    # Add the image to the canvas
                    new_id = self.add_image(
                        img,
                        parent_id=parent_id,
                        pos=(x, y),
                        image_type=image_type,
                        filename=node_data['filename']
                    )
                    
                    # Set the name if provided
                    if name:
                        self.images[new_id].set_name(name)
                        
                    # Store the ID mapping
                    id_mapping[node_data['id']] = new_id
                    
                    # Load children recursively
                    if 'children' in node_data:
                        for child_data in node_data['children']:
                            load_node(child_data, parent_id=new_id)
                            
                    return new_id
                else:
                    print(f"Error: Node has no filename and no valid function name: {name}")
                    
                return None
                
            # Load the tree starting from the root
            root_id = load_node(tree_data)
            
            if not show_solution:
                # Process the last image to result
                self.process_last_image_to_result()
            
            # Start relaxation to adjust positions
            self.start_relaxation()
            
        except Exception as e:
            print(f"Error loading YAML file: {e}")

    def process_last_image_to_result(self):
        """Process the last image in the canvas to create a result image.
        This method will:
        1. Keep all existing images
        2. Find the last image added to the canvas
        3. Create a new image with the same pixels as the last image, but with a new name "result"
        4. Update the result image if it already exists
        """
        # Find the last image ID (excluding ID 0)
        last_image_id = None
        for img_id in self.images:
            if img_id != 0 and (last_image_id is None or img_id > last_image_id):
                last_image_id = img_id
                
        if last_image_id is not None:
            # Get the last image's data
            last_image = self.images[last_image_id]
            last_image_array = last_image.get_array()
            last_image_pos = last_image.get_pos()
            last_image_type = last_image.get_image_type()
            
            # Check if we already have a result image
            result_id = None
            for img_id, img in self.images.items():
                if img.get_name() == "result":
                    result_id = img_id
                    break
            
            if result_id is None:
                # Create a new result image if we don't have one
                result_id = self.add_image(
                    last_image_array,
                    parent_id=None,  # No parent relationship
                    pos=last_image_pos,
                    image_type=last_image_type,
                    name="result"
                )
                self.move_result_to_bottom_right()
            else:
                # Update the existing result image
                self.images[result_id].array = last_image_array
                self.images[result_id].image_type = last_image_type
                # Update the surface for display
                if last_image_type == 'label':
                    display_array = self.images[result_id]._label_to_rgb(last_image_array)
                else:
                    if last_image_array.max() <= 1.0:
                        display_array = last_image_array * 255
                    else:
                        display_array = last_image_array
                    if len(display_array.shape) == 2:
                        display_array = np.stack([display_array] * 3, axis=-1)
                display_array = display_array.astype(np.uint8)
                self.images[result_id].surface = pygame.surfarray.make_surface(display_array)
                self.move_result_to_bottom_right()
                
            # Remove all derived images after processing to result
            self.remove_derived_images()
            
    def remove_derived_images(self):
        """Remove all images derived from the original image (ID 0) except the result image."""
        print("Removing derived images")
        # Get all images that are derived from the original image
        derived_images = []
        for img_id, img in self.images.items():
            # Skip the original image (ID 0) and the result image
            if img_id == 0 or self.is_result_image(img_id):
                continue
                
            # Check if this image is derived from the original image
            if self.is_derived_from_original(img_id):
                derived_images.append(img_id)
                
        # Remove all derived images
        for img_id in derived_images:
            if img_id in self.images:
                del self.images[img_id]
                
        # Start relaxation to adjust remaining images
        self.start_relaxation()
        
    def is_derived_from_original(self, img_id, visited=None):
        """Check if an image is derived from the original image (ID 0).
        
        Args:
            img_id: ID of the image to check
            visited: Set of already visited image IDs to prevent infinite recursion
            
        Returns:
            bool: True if the image is derived from the original image, False otherwise
        """
        # Initialize visited set if not provided
        if visited is None:
            visited = set()
            
        # If we've already visited this image, return False to prevent infinite recursion
        if img_id in visited:
            return False
            
        # Mark this image as visited
        visited.add(img_id)
        
        # If this is the original image (ID 0), return True
        if img_id == 0:
            return True
            
        # Get the parent image
        parent_id = self.images[img_id].get_parent_id()
        if parent_id is not None and parent_id in self.images:
            # Recursively check if the parent is derived from the original image
            return self.is_derived_from_original(parent_id, visited)
            
        # If no parent found, return False
        return False

    def render_level_buttons(self):
        """Render the level button and level selection buttons if visible."""
        # Render main level button
        button_color = self.level_button_hover_color if self.level_button_hovered else self.level_button_color
        pygame.draw.rect(self.screen, button_color, self.level_button_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), self.level_button_rect, 2)
        
        text_surface = self.save_button_font.render(self.level_button_text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.level_button_rect.center)
        self.screen.blit(text_surface, text_rect)
        
        # Render level selection buttons if visible
        if self.level_buttons_visible:
            for i, button_rect in enumerate(self.level_buttons, 1):
                pygame.draw.rect(self.screen, self.level_button_color, button_rect)
                pygame.draw.rect(self.screen, (255, 255, 255), button_rect, 2)
                
                text_surface = self.save_button_font.render(str(i), True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=button_rect.center)
                self.screen.blit(text_surface, text_rect)

    def handle_level_button_click(self, pos):
        """Handle clicks on the level button and level selection buttons."""
        # Check if main level button was clicked
        if self.level_button_rect.collidepoint(pos):
            self.level_buttons_visible = not self.level_buttons_visible
            return True
            
        # Check if any level selection button was clicked
        if self.level_buttons_visible:
            for i, button_rect in enumerate(self.level_buttons, 1):
                if button_rect.collidepoint(pos):
                    # Load the corresponding level file
                    level_file = f"level{i}.yaml"
                    try:
                        self.load_yaml_tree(level_file)
                    except Exception as e:
                        print(f"Error loading level file {level_file}: {e}")
                    self.level_buttons_visible = False
                    return True
                    
        # If clicked outside level buttons, hide them
        if self.level_buttons_visible:
            self.level_buttons_visible = False
            return True
            
        return False

    def handle_solution_button_click(self, pos):
        """Handle clicks on the solution button."""
        # Check if solution button was clicked
        if self.solution_button_rect.collidepoint(pos):
            self.load_yaml_tree(f"level{self.current_level}.yaml", show_solution=True)
            return True

    def is_result_image(self, img_id):
        """Check if an image is a result image.
        
        Args:
            img_id: ID of the image to check
            
        Returns:
            bool: True if the image is a result image, False otherwise
        """
        if img_id in self.images:
            return self.images[img_id].get_name() == "result"
        return False

    def move_result_to_bottom_right(self):
        """Move the result image to the bottom right corner of the canvas."""
        # Find the result image
        result_id = None
        for img_id, img in self.images.items():
            if self.is_result_image(img_id):
                result_id = img_id
                break
                
        if result_id is not None:
            # Calculate position in bottom right corner with margin
            # Scale the margin with the current zoom level to maintain consistent visual distance
            scaled_margin = img.radius

            # Calculate position in bottom right corner
            new_pos = (self.width - scaled_margin, self.height - scaled_margin)
            
            # Update the result image's position
            self.images[result_id].set_pos(new_pos)
            return True
        return False

    def compute_quality_metric(self, img1_array, img2_array, img1_type, img2_type):
        """Compute quality metric between two images.
        
        Args:
            img1_array: First image array
            img2_array: Second image array
            img1_type: Type of first image ('intensity', 'binary', or 'label')
            img2_type: Type of second image ('intensity', 'binary', or 'label')
            
        Returns:
            tuple: (metric_name, metric_value) or (None, None) if types don't match
        """
        if img1_type != img2_type:
            return None, None
            
        if img1_type == 'intensity':
            # Compute MSE using clesperanto
            import pyclesperanto as cle
            img1_cle = cle.push(img1_array)
            img2_cle = cle.push(img2_array)
            mse = cle.mean_squared_error(img1_cle, img2_cle)
            return "MSE:", mse
        elif img1_type in ['binary', 'label']:
            # Compute Jaccard index
            from sklearn.metrics import confusion_matrix
            import numpy as np
            
            img1_array = np.asarray(img1_array)
            img2_array = np.asarray(img2_array)
            
            # determine overlap
            overlap = confusion_matrix(img2_array.ravel(), img1_array.ravel())
            
            # crop out region in confusion matrix where reference labels are
            num_labels_reference = int(img1_array.max())
            overlap = overlap[0:num_labels_reference+1, :]
            
            # Measure correctly labeled pixels
            n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
            n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
            
            # Calculate intersection over union
            divisor = (n_pixels_pred + n_pixels_true - overlap)
            is_zero = divisor == 0
            divisor[is_zero] = 1
            overlap[is_zero] = 0
            iou = overlap / divisor
            
            # ignore background
            iou = iou[1:,1:]
            
            max_jacc = iou.max(axis=1)
            
            quality = max_jacc.mean()
            
            return "IoU:", quality
            
        return None, None

    def render_quality_metric(self):
        """Render the quality metric on the screen."""
        if self.quality_metric is not None:
            metric_name, metric_value = self.quality_metric
            if metric_name and metric_value is not None:
                # Format the metric value
                if metric_name == "MSE:":
                    formatted_value = f"{metric_value:.6f}"
                else:  # IoU
                    formatted_value = f"{metric_value:.4f}"
                    
                # Create the text
                text = f"{metric_name} {formatted_value}"
                text_surface = self.quality_metric_font.render(text, True, (255, 255, 255))
                
                # Draw background
                pygame.draw.rect(self.screen, (0, 0, 0, 128), self.quality_metric_rect)
                
                # Draw text
                text_rect = text_surface.get_rect(center=self.quality_metric_rect.center)
                self.screen.blit(text_surface, text_rect)

    def check_quality_and_load_next_level(self):
        """Check if quality metrics meet thresholds and load next level if they do."""
        if self.quality_metric is None:
            return
            
        metric_name, metric_value = self.quality_metric
        
        # Check if metrics meet thresholds
        if metric_name == "MSE:" and metric_value < 0.001:
            self.level_completed = True
            self.level_completion_time = pygame.time.get_ticks()
            self.quality_metric = None
        
        elif metric_name == "IoU:" and metric_value > 0.999:
            self.level_completed = True
            self.level_completion_time = pygame.time.get_ticks()
            self.quality_metric = None
        
        
        
            
    def load_next_level(self):
        """Load the next level if available."""
        next_level = self.current_level + 1
        level_file = os.path.join(os.path.dirname(__file__), f"data/level{next_level}.yaml")
        
        # Check if next level file exists
        if os.path.exists(level_file):
            try:
                self.load_yaml_tree(level_file)
                print(f"Loading next level: {next_level}")
            except Exception as e:
                print(f"Error loading next level {level_file}: {e}")
        else:
            print("Congratulations! You've completed all levels!")

    def render_solution_button(self):
        """Render the solution button on the screen."""
        button_color = self.solution_button_hover_color if self.solution_button_hovered else self.solution_button_color
        pygame.draw.rect(self.screen, button_color, self.solution_button_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), self.solution_button_rect, 2)
        
        text_surface = self.save_button_font.render(self.solution_button_text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=self.solution_button_rect.center)
        self.screen.blit(text_surface, text_rect)


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
   
    def split_touching_objects(binary, sigma: float = 3.5):
        """
        Takes a binary image and draws cuts in the objects similar to the ImageJ watershed algorithm [1].
        """
        binary = np.asarray(binary, dtype=np.uint8)

        # typical way of using scikit-image watershed
        distance = ndi.distance_transform_edt(binary)
        blurred_distance = gaussian(distance, sigma=sigma)
        fp = np.ones((3,) * binary.ndim)
        coords = peak_local_max(blurred_distance, footprint=fp, labels=binary)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers = label(mask)
        labels = watershed(-blurred_distance, markers, mask=binary)

        # identify label-cutting edges
        edges = sobel(labels)
        edges2 = sobel(binary)
        
        almost = np.logical_not(np.logical_xor(edges != 0, edges2 != 0)) * binary
        return sk_binary_opening(almost) * 1
    
    def local_minima_seeded_watershed(image, spot_sigma: float = 10, outline_sigma: float = 0):
        """
        Segment cells in images with fluorescently marked membranes.

        The two sigma parameters allow tuning the segmentation result. The first sigma controls how close detected cells
        can be (spot_sigma) and the second controls how precise segmented objects are outlined (outline_sigma). Under the
        hood, this filter applies two Gaussian blurs, local minima detection and a seeded watershed.

        See also
        --------
        .. [1] https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
        """

        image = np.asarray(image)

        spot_blurred = gaussian(image, sigma=spot_sigma)

        spots = label(local_minima(spot_blurred))

        if outline_sigma == spot_sigma:
            outline_blurred = spot_blurred
        else:
            outline_blurred = gaussian(image, sigma=outline_sigma)

        return watershed(outline_blurred, spots)

    def distance_map(binary):
        binary = np.asarray(binary, dtype=np.uint8)
        dm = ndi.distance_transform_edt(binary)
        dm = dm / dm.max() * 255
        return dm.astype(np.uint8)  

    # Functions for intensity images
    intensity_functions = {
        'filter': {
            'gaussian': lambda img: to_numpy(cle.gaussian_blur(to_cle(img), sigma_x=2, sigma_y=2)),
            'median': lambda img: to_numpy(cle.median_box(to_cle(img), radius_x=2, radius_y=2)),
            'top_hat': lambda img: to_numpy(cle.top_hat_box(to_cle(img), radius_x=5, radius_y=5)),
            'laplace': lambda img: to_numpy(cle.laplace(to_cle(img))),
            'minimum': lambda img: to_numpy(cle.minimum_box(to_cle(img), radius_x=2, radius_y=2)),
            'maximum': lambda img: to_numpy(cle.maximum_box(to_cle(img), radius_x=2, radius_y=2)),
            'mean': lambda img: to_numpy(cle.mean_box(to_cle(img), radius_x=2, radius_y=2)),
            'variance': lambda img: to_numpy(cle.variance_box(to_cle(img), radius_x=2, radius_y=2)),
            'invert': lambda img: to_numpy(np.max(img) - img)
        },
        'binarization': {
            'otsu': lambda img: to_numpy(cle.threshold_otsu(to_cle(img))),
            'morphological_chan_vese': lambda img: to_numpy(cle.morphological_chan_vese(to_cle(img), num_iter=10)),
            'detect_maxima': lambda img: to_numpy(cle.detect_maxima(to_cle(img), radius_x=2, radius_y=2))
        },
        'label': {
            'watershed': local_minima_seeded_watershed,
            'voronoi-otsu': lambda img: to_numpy(cle.voronoi_otsu_labeling(to_cle(img), spot_sigma=2, outline_sigma=2))
        },
        'background': {
            'subtract_gaussian': lambda img: to_numpy(cle.subtract_gaussian_background(to_cle(img), sigma_x=10, sigma_y=10)),
            'divide_by_gaussian': lambda img: to_numpy(cle.divide_by_gaussian_background(to_cle(img), sigma_x=10, sigma_y=10))
        }
    }
    
    # Functions for binary images
    binary_functions = {
        'filter': {
            'minimum': lambda img: to_numpy(cle.minimum_box(to_cle(img), radius_x=2, radius_y=2)),
            'maximum': lambda img: to_numpy(cle.maximum_box(to_cle(img), radius_x=2, radius_y=2)),
            'dilate': lambda img: to_numpy(cle.maximum_box(to_cle(img), radius_x=2, radius_y=2)),
            'erode': lambda img: to_numpy(cle.minimum_box(to_cle(img), radius_x=2, radius_y=2)),
            'open': lambda img: to_numpy(cle.opening_box(to_cle(img), radius_x=2, radius_y=2)),
            'close': lambda img: to_numpy(cle.closing_box(to_cle(img), radius_x=2, radius_y=2)),
            'binary_not': lambda img: to_numpy(cle.binary_not(to_cle(img))),
            'binary_edge': lambda img: to_numpy(cle.binary_edge_detection(to_cle(img))),
            'split_touching_objects': split_touching_objects,
            'distance_map': distance_map
        },
        'label': {
            'connected_components': lambda img: to_numpy(cle.connected_component_labeling(to_cle(img))).astype(np.int32),
            'label_spots': lambda img: to_numpy(cle.label_spots(to_cle(img)))
        }
    }
    
    # Functions for labeled images
    label_functions = {
        'filter': {
            'erode_labels': lambda img: to_numpy(cle.erode_labels(to_cle(img), radius=2)),
            'dilate_labels': lambda img: to_numpy(cle.dilate_labels(to_cle(img), radius=2)),
            'smooth_labels': lambda img: to_numpy(cle.smooth_labels(to_cle(img), radius=5)),
            'extend_via_voronoi': lambda img: to_numpy(cle.extend_labeling_via_voronoi(to_cle(img))),
            'remove_small': lambda img: to_numpy(cle.remove_small_labels(to_cle(img), minimum_size=100)),
            'remove_large': lambda img: to_numpy(cle.remove_large_labels(to_cle(img), maximum_size=1000)),
            'mode': lambda img: to_numpy(cle.mode_box(to_cle(img), radius_x=2, radius_y=2))
        },
        'reduce': {
            'centroids': lambda img: to_numpy(cle.reduce_labels_to_centroids(to_cle(img))),
            'outlines': lambda img: to_numpy(cle.reduce_labels_to_label_edges(to_cle(img)))
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

def filter_functions(categories, valid_names):
    """Filter functions based on valid names."""
    if valid_names is None:
        return categories
    
    print("valid_names", valid_names)

    filtered_categories = {}
    for c_name, functions in categories.items():
        filtered_functions = {}
        for name, function in functions.items():
            if name in valid_names:
                filtered_functions[name] = function
            else:
                print("ignoring", name)
        if len(filtered_functions) > 0:
            filtered_categories[c_name] = filtered_functions
        
    return filtered_categories


def main():
    # Create canvas with our enhanced function collection
    canvas = ImageProcessingCanvas(800, 600, create_dummy_functions())
    
    # Load level1.yaml instead of an image
    canvas.load_yaml_tree(os.path.join(os.path.dirname(__file__), f"data/level1.yaml"))
    
    # Run
    canvas.run()


# Demo
if __name__ == "__main__":
    main()
    
