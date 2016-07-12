/**
* This file is part of RGBID-SLAM.
*
* Copyright (C) 2015 Daniel Gutiérrez Gómez <danielgg at unizar dot es> (Universidad de Zaragoza)
*
* RGBID-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* RGBID-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with RGBID-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include <settings.h>
#include <fstream>

RGBID_SLAM::Entry::Entry(std::string name, std::string value):name_(name),value_(value)
{}

RGBID_SLAM::Section::Section(std::string name):name_(name)
{}
    
void RGBID_SLAM::Section::addEntry(Entry &new_entry)
{      
  if (entries_.find(new_entry.getName()) != entries_.end())
  {
    std::cout << "Warning: entry " << new_entry.getName() << " is already loaded." << std::endl;
    return;
  }
  
  entries_[new_entry.getName()] = new_entry;     
}

bool RGBID_SLAM::Section::getEntry(const std::string &entry_name, Entry &entry) const
{
  std::map<std::string, Entry>::const_iterator it;
  
  it = entries_.find(entry_name);
  if (it != entries_.end())
  {
    entry = (it)->second;
    return  true;
  }
  else
  {
    return false;
  }
}

RGBID_SLAM::Settings::Settings (std::ifstream& filestream)
{
  load(filestream);
}
    
void RGBID_SLAM::Settings::load (std::ifstream& filestream)
{  
  std::string entry_name;
  std::string entry_value;
  std::string section_name;
  std::string line;
  
  while (std::getline(filestream,line))
  {
    line = trim(line);
    
    if (!line.length()) 
      continue;
    
    if (line[0] == '#') 
      continue;
      
    if (line[0] == ';') 
      continue;
    
    if (line[0] == '[')
    {
      section_name = trim(line.substr(1,line.find(']')-1));
      
      Section new_section(section_name);
      
      addSection(new_section);
      
      continue;
    }
    
    int pos_equal = line.find('=');
    
    if (pos_equal != std::string::npos)
    {
      entry_name = trim(line.substr(0,pos_equal));
      entry_value = trim(line.substr(pos_equal+1));
      
      if (!entry_name.empty())
      {
        Entry new_entry(entry_name, entry_value);
    
        sections_[section_name].addEntry(new_entry); 
      }        
    }
    else
    {
      if (!entry_name.empty())
      {
        entry_value+='\n';
        entry_value+=trim(line);
        sections_[section_name].entries_[entry_name].setValue(entry_value);
      } 
    }
    
    //////////////////////////
    //if (pos_equal == std::string::npos)
      //continue;
      
    //entry_name = trim(line.substr(0,pos_equal));
    //entry_value = trim(line.substr(pos_equal+1));
    
    //if (entry_name.empty() || entry_value.empty())
      //continue;
    
    //Entry new_entry(entry_name, entry_value);
    
    //sections_[section_name].addEntry(new_entry);    
  }
  
  for (std::map<std::string, Section>::iterator it = sections_.begin(); it != sections_.end(); it++)
  {
    std::cout << (it->second).getName() << std::endl;
    
    for (std::map<std::string, Entry>::iterator it_e = (it->second).entries_.begin(); it_e != (it->second).entries_.end(); it_e++)
    {
      std::cout << "    " << (it_e->second).getName() << ": " << (it_e->second).getValue() << std::endl;
    }
  }
};

void RGBID_SLAM::Settings::addSection(Section &new_section)
{
  if (sections_.find(new_section.getName()) != sections_.end())
  {
    std::cout << "Warning: section " << new_section.getName() << " is already loaded." << std::endl;
    return;
  }
  
  sections_[new_section.getName()] = new_section;
}


bool RGBID_SLAM::Settings::getSection(const std::string &section_name, Section& section) const
{
  std::map<std::string, Section>::const_iterator it;
  
  it = sections_.find(section_name);
  if (it != sections_.end())
  {
    section = (it)->second;
    return true;
  }
  else
  {
    return false;
  }
}
    
  
